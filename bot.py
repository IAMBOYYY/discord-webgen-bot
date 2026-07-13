import os
import json
import asyncio
import threading
import shutil
import uuid
import time
import base64
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import discord
from discord.ext import commands
from openai import AsyncOpenAI, RateLimitError, APIStatusError
import aiohttp
import aiofiles

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
PORT = int(os.getenv("PORT", 8000))
WORKSPACE_DIR = Path("workspace")
WORKSPACE_DIR.mkdir(exist_ok=True)
META_FILE = WORKSPACE_DIR / "workspace_meta.json"
PROJECT_TTL_HOURS = 2
ADMIN_PASSWORD = "26jan24march"

# Provider defaults
PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "google/gemma-2-9b-it",
    },
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.1-8b-instant",
    },
    "nvidia": {
        "name": "NVIDIA NIM",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": "meta/llama-3.1-8b-instruct",
    },
    "mistral": {
        "name": "Mistral AI",
        "base_url": "https://api.mistral.ai/v1",
        "default_model": "mistral-small-latest",
    },
    "cerebras": {
        "name": "Cerebras",
        "base_url": "https://api.cerebras.ai/v1",
        "default_model": "llama3.1-8b",
    },
    "vertex": {
        "name": "Vertex AI",
        "base_url": os.getenv("VERTEX_BASE_URL", ""),  # optional, leave empty to ignore
        "default_model": "gemini-1.5-flash",
    },
}

# -------------------------------------------------------------------
# User data
# -------------------------------------------------------------------
USER_DATA_FILE = Path("user_data.json")
user_data = {}   # user_id(str) -> { "provider": "...", "api_key": "...", "model": "...", "current_project": "..." }

def load_user_data():
    global user_data
    if USER_DATA_FILE.exists():
        with open(USER_DATA_FILE, "r") as f:
            user_data = json.load(f)
    else:
        user_data = {}

def save_user_data():
    with open(USER_DATA_FILE, "w") as f:
        json.dump(user_data, f, indent=2)

load_user_data()

# -------------------------------------------------------------------
# Workspace meta
# -------------------------------------------------------------------
workspace_meta = {}

def load_workspace_meta():
    global workspace_meta
    if META_FILE.exists():
        with open(META_FILE, "r") as f:
            workspace_meta = json.load(f)
    else:
        workspace_meta = {}

def save_workspace_meta():
    with open(META_FILE, "w") as f:
        json.dump(workspace_meta, f, indent=2)

load_workspace_meta()

# -------------------------------------------------------------------
# HTTP server
# -------------------------------------------------------------------
class WorkspaceHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WORKSPACE_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            super().do_GET()

def start_http_server():
    server = HTTPServer(("0.0.0.0", PORT), WorkspaceHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------
async def delete_project(project_folder: str):
    path = WORKSPACE_DIR / project_folder
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    if project_folder in workspace_meta:
        del workspace_meta[project_folder]
        save_workspace_meta()

async def schedule_project_deletion(project_folder: str, delay_hours: float = PROJECT_TTL_HOURS):
    await asyncio.sleep(delay_hours * 3600)
    await delete_project(project_folder)

async def startup_cleanup():
    now = time.time()
    expired = [f for f, ts in workspace_meta.items() if (now - ts) > PROJECT_TTL_HOURS * 3600]
    for f in expired:
        await delete_project(f)
    for f, ts in workspace_meta.items():
        remaining = PROJECT_TTL_HOURS * 3600 - (now - ts)
        if remaining > 0:
            asyncio.create_task(schedule_project_deletion(f, remaining / 3600))
        else:
            await delete_project(f)

# -------------------------------------------------------------------
# Discord bot
# -------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
intents.members = True   # Enable in Developer Portal for online member list
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")
    start_http_server()
    await startup_cleanup()

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def get_user_client(user_id: str):
    uid = str(user_id)
    if uid not in user_data or "api_key" not in user_data[uid]:
        return None
    info = user_data[uid]
    provider = info.get("provider", "openrouter")
    prov_cfg = PROVIDERS.get(provider)
    if not prov_cfg or not prov_cfg["base_url"]:
        return None
    model = info.get("model", prov_cfg["default_model"])
    return AsyncOpenAI(api_key=info["api_key"], base_url=prov_cfg["base_url"])

def get_user_model(user_id: str) -> str:
    uid = str(user_id)
    if uid in user_data and "model" in user_data[uid]:
        return user_data[uid]["model"]
    provider = user_data.get(uid, {}).get("provider", "openrouter")
    return PROVIDERS[provider]["default_model"]

# -------------------------------------------------------------------
# Fetch models
# -------------------------------------------------------------------
async def fetch_models(provider: str, api_key: str) -> list:
    prov = PROVIDERS.get(provider)
    if not prov or not prov["base_url"]:
        return []
    url = f"{prov['base_url']}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                models = data.get("data", data.get("models", []))
                return [m["id"] for m in models if "id" in m]
        except Exception:
            return []

# -------------------------------------------------------------------
# Show all models in paginated messages
# -------------------------------------------------------------------
async def show_all_models(ctx, models_list: list):
    """Send all model IDs in batches, each message contains up to 50 models with global indices."""
    if not models_list:
        return await ctx.send("No models found.")
    total = len(models_list)
    # Create a single large text block, then split if needed
    lines = []
    for i, model_id in enumerate(models_list, 1):
        lines.append(f"`{i}` - `{model_id}`")
    full_text = "\n".join(lines)
    # Discord message limit is 2000 chars; if we exceed, send in chunks
    if len(full_text) <= 2000:
        await ctx.send(f"**All {total} models:**\n{full_text}")
    else:
        # Split into chunks of 50 lines
        chunk_size = 50
        chunks = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
        for idx, chunk in enumerate(chunks):
            start_num = idx * chunk_size + 1
            end_num = min(start_num + chunk_size - 1, total)
            header = f"**Models {start_num}–{end_num} of {total}:**"
            await ctx.send(header + "\n" + "\n".join(chunk))
            await asyncio.sleep(0.5)  # avoid rate limits

# -------------------------------------------------------------------
# Interactive setup (all models, 5 min timeout)
# -------------------------------------------------------------------
@bot.command(name="setup")
async def setup(ctx):
    """Guide through choosing provider, setting API key, and picking a model."""
    uid = str(ctx.author.id)
    lines = ["**Choose your AI provider:**"]
    keys = list(PROVIDERS.keys())
    for i, key in enumerate(keys, 1):
        lines.append(f"`{i}` - {PROVIDERS[key]['name']}")
    lines.append("Reply with the number (e.g., `1`).")
    await ctx.send("\n".join(lines))

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel and m.content.isdigit()

    try:
        msg = await bot.wait_for("message", timeout=120.0, check=check)
        choice = int(msg.content)
        if choice < 1 or choice > len(keys):
            return await ctx.send("Invalid choice. Run `!setup` again.")
        selected_provider = keys[choice - 1]
        prov_cfg = PROVIDERS[selected_provider]

        # Vertex requires base URL (optional, you can skip)
        if selected_provider == "vertex":
            await ctx.send("Enter your Vertex AI base URL (e.g., https://...):")
            url_msg = await bot.wait_for("message", timeout=120.0, check=lambda m: m.author == ctx.author and m.channel == ctx.channel)
            PROVIDERS["vertex"]["base_url"] = url_msg.content.strip()
            prov_cfg["base_url"] = url_msg.content.strip()

        await ctx.send(f"✅ Selected **{prov_cfg['name']}**. Now send me your API key (use a DM for safety). You have 5 minutes.")
        msg = await bot.wait_for("message", timeout=300.0, check=lambda m: m.author == ctx.author and m.channel == ctx.channel)
        api_key = msg.content.strip()
        if ctx.guild:
            try: await msg.delete()
            except: pass

        await ctx.send("🔍 Fetching all available models for your API key…")
        models = await fetch_models(selected_provider, api_key)
        if not models:
            default_model = prov_cfg["default_model"]
            user_data[uid] = {"provider": selected_provider, "api_key": api_key, "model": default_model, "current_project": None}
            save_user_data()
            return await ctx.send(f"⚠️ Could not fetch model list. Using default model: `{default_model}`. Change later with `!models`.")

        models_sorted = sorted(models)
        # Show every model (paginated)
        await show_all_models(ctx, models_sorted)
        await ctx.send("Reply with the number from the list or the exact model ID. You have 5 minutes.")

        def model_check(m):
            return m.author == ctx.author and m.channel == ctx.channel

        msg = await bot.wait_for("message", timeout=300.0, check=model_check)
        choice_text = msg.content.strip()
        chosen_model = None
        if choice_text.isdigit():
            idx = int(choice_text)
            if 1 <= idx <= len(models_sorted):
                chosen_model = models_sorted[idx - 1]
            else:
                return await ctx.send("Invalid number. Setup cancelled. Run `!setup` again.")
        else:
            if choice_text in models_sorted:
                chosen_model = choice_text
            else:
                return await ctx.send("Model ID not found. Setup cancelled.")

        user_data[uid] = {"provider": selected_provider, "api_key": api_key, "model": chosen_model, "current_project": None}
        save_user_data()
        await ctx.send(f"✅ Setup complete!\nProvider: **{prov_cfg['name']}**\nModel: `{chosen_model}`")

    except asyncio.TimeoutError:
        await ctx.send("⌛ Setup timed out. Run `!setup` again.")

# -------------------------------------------------------------------
# !models – show all models and let user switch (5 min timeout)
# -------------------------------------------------------------------
@bot.command(name="models")
async def list_models(ctx):
    """List all available models for your current provider and allow switching."""
    uid = str(ctx.author.id)
    if uid not in user_data or "api_key" not in user_data[uid]:
        return await ctx.send("❌ You haven't set up a provider and API key. Use `!setup`.")
    provider = user_data[uid]["provider"]
    api_key = user_data[uid]["api_key"]
    await ctx.send("🔍 Fetching all models…")
    models = await fetch_models(provider, api_key)
    if not models:
        return await ctx.send("❌ Could not retrieve models. Check your API key.")
    models_sorted = sorted(models)
    await show_all_models(ctx, models_sorted)
    await ctx.send("Reply with the number from the list or the exact model ID to switch, or `cancel`. You have 5 minutes.")

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel
    try:
        msg = await bot.wait_for("message", timeout=300.0, check=check)
        choice = msg.content.strip()
        if choice.lower() == "cancel":
            return await ctx.send("Cancelled.")
        chosen = None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(models_sorted):
                chosen = models_sorted[idx - 1]
        else:
            if choice in models_sorted:
                chosen = choice
        if chosen:
            user_data[uid]["model"] = chosen
            save_user_data()
            await ctx.send(f"✅ Model switched to `{chosen}`.")
        else:
            await ctx.send("Invalid selection.")
    except asyncio.TimeoutError:
        await ctx.send("⌛ Timed out.")

# -------------------------------------------------------------------
# !setkey, !provider (unchanged)
# -------------------------------------------------------------------
@bot.command(name="setkey")
async def set_key(ctx, *, key: str):
    uid = str(ctx.author.id)
    if uid not in user_data: return await ctx.send("❌ Run `!setup` first.")
    user_data[uid]["api_key"] = key.strip()
    save_user_data()
    if ctx.guild:
        try: await ctx.message.delete()
        except: pass
        await ctx.send("✅ API key saved.", delete_after=10)
    else:
        await ctx.send("✅ API key saved.")

@bot.command(name="provider")
async def show_provider(ctx):
    uid = str(ctx.author.id)
    if uid in user_data:
        p = user_data[uid]["provider"]
        m = get_user_model(uid)
        proj = user_data[uid].get("current_project", "none")
        await ctx.send(f"Provider: **{PROVIDERS[p]['name']}**\nModel: `{m}`\nActive project: `{proj}`")
    else:
        await ctx.send("Not set up.")

# -------------------------------------------------------------------
# !ask – server aware, shows online member names (intent needed)
# -------------------------------------------------------------------
SERVER_KEYWORDS = [
    "server", "guild", "member", "owner", "moderator", "admin",
    "channel", "role", "emojis", "boost", "level", "created",
    "this place", "here", "how many people", "who owns", "who is online", "online members"
]

def is_server_question(q: str) -> bool:
    return any(word in q.lower() for word in SERVER_KEYWORDS)

def get_server_info(guild: discord.Guild) -> str:
    total = guild.member_count
    online_members = [m.name for m in guild.members if m.status != discord.Status.offline]
    online_str = ", ".join(online_members[:10])
    if len(online_members) > 10:
        online_str += f" and {len(online_members)-10} more"
    text_ch = len(guild.text_channels)
    voice_ch = len(guild.voice_channels)
    roles = ", ".join([r.name for r in guild.roles[:10]])
    owner = guild.owner.name if guild.owner else "Unknown"
    created = guild.created_at.strftime("%B %d, %Y")
    return (
        f"Server name: {guild.name}\n"
        f"Owner: {owner}\n"
        f"Total members: {total}\n"
        f"Online members ({len(online_members)}): {online_str}\n"
        f"Text channels: {text_ch}, Voice channels: {voice_ch}\n"
        f"Roles (sample): {roles}\n"
        f"Created on: {created}"
    )

@bot.command(name="ask")
async def ask_question(ctx, *, question: str):
    uid = str(ctx.author.id)
    client = get_user_client(uid)
    if not client: return await ctx.send("❌ Set up provider & key first.")
    model = get_user_model(uid)
    messages = []
    if ctx.guild and is_server_question(question):
        server_info = get_server_info(ctx.guild)
        system_content = f"You have real-time server data:\n{server_info}\nAnswer using this."
    else:
        system_content = "You are a helpful assistant. Answer the user's question."
    messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": question})
    async with ctx.typing():
        try:
            resp = await client.chat.completions.create(model=model, messages=messages, temperature=0.7, max_tokens=1024)
            await ctx.send(resp.choices[0].message.content[:2000])
        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")

# -------------------------------------------------------------------
# !newproject, !listfiles, !viewfile, !setproject, !listprojects
# -------------------------------------------------------------------
@bot.command(name="newproject")
async def new_project(ctx):
    uid = str(ctx.author.id)
    proj_name = f"proj_{uuid.uuid4().hex[:8]}"
    (WORKSPACE_DIR / proj_name).mkdir(parents=True, exist_ok=True)
    workspace_meta[proj_name] = time.time()
    save_workspace_meta()
    if uid not in user_data: user_data[uid] = {}
    user_data[uid]["current_project"] = proj_name
    save_user_data()
    asyncio.create_task(schedule_project_deletion(proj_name))
    render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
    await ctx.send(f"📁 New project: `{proj_name}`\n🌐 {render_url}/{proj_name}/")

@bot.command(name="listfiles")
async def list_files_cmd(ctx):
    uid = str(ctx.author.id)
    proj = user_data.get(uid, {}).get("current_project")
    if not proj: return await ctx.send("No active project.")
    proj_path = WORKSPACE_DIR / proj
    files = [str(p.relative_to(proj_path)) for p in proj_path.rglob("*") if p.is_file()]
    if files:
        await ctx.send("**Files:**\n" + "\n".join(f"• `{f}`" for f in files))
    else:
        await ctx.send("No files yet.")

@bot.command(name="viewfile")
async def view_file(ctx, filename: str):
    uid = str(ctx.author.id)
    proj = user_data.get(uid, {}).get("current_project")
    if not proj: return await ctx.send("No active project.")
    file_path = WORKSPACE_DIR / proj / filename
    if not file_path.exists(): return await ctx.send("File not found.")
    async with aiofiles.open(file_path, "r") as f:
        content = await f.read()
    if len(content) > 2000:
        await ctx.send(file=discord.File(file_path, filename=filename))
    else:
        await ctx.send(f"**{filename}**\n```{filename.split('.')[-1]}\n{content}\n```")

@bot.command(name="setproject")
async def set_project(ctx, project_name: str):
    uid = str(ctx.author.id)
    if not (WORKSPACE_DIR / project_name).exists(): return await ctx.send("Project not found.")
    user_data[uid]["current_project"] = project_name
    save_user_data()
    await ctx.send(f"✅ Active project: `{project_name}`")

@bot.command(name="listprojects")
async def list_projects(ctx):
    render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
    if not workspace_meta: return await ctx.send("No projects.")
    lines = ["**Projects:**"]
    for folder in workspace_meta:
        lines.append(f"• `{folder}` → {render_url}/{folder}/")
    await ctx.send("\n".join(lines))

# -------------------------------------------------------------------
# !make (multi-language, Unsplash images, multi-page)
# -------------------------------------------------------------------
@bot.command(name="make")
async def make_website(ctx, *, description: str):
    uid = str(ctx.author.id)
    if uid not in user_data or not user_data[uid].get("current_project"):
        return await ctx.send("❌ No active project. Use `!newproject`.")
    client = get_user_client(uid)
    if not client: return await ctx.send("❌ Set up provider & key first.")
    model = get_user_model(uid)
    proj_name = user_data[uid]["current_project"]
    proj_path = WORKSPACE_DIR / proj_name

    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_file",
                "description": "Create a new file with given content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Relative path, e.g., 'index.html'"},
                        "content": {"type": "string", "description": "Full file content"},
                    },
                    "required": ["filename", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file's content.",
                "parameters": {
                    "type": "object",
                    "properties": {"filename": {"type": "string"}},
                    "required": ["filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List all files in project.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "Delete a file.",
                "parameters": {
                    "type": "object",
                    "properties": {"filename": {"type": "string"}},
                    "required": ["filename"],
                },
            },
        },
    ]

    system_prompt = (
        "You are an expert software developer. Create the project according to the user's request.\n\n"
        "CRITICAL RULES:\n"
        "- Use tools to create, read, or delete files.\n"
        "- For **websites**, always create separate HTML, CSS, JS files. Make it fully responsive and beautiful. "
        "Use modern CSS with animations. For images, use Unsplash URLs like `https://source.unsplash.com/random/800x600/?topic`.\n"
        "- For **games** or **interactive projects**, ensure all logic works.\n"
        "- For **non-web** projects (Python, Java, etc.), create the appropriate code files (e.g., `.py`, `.java`).\n"
        "- If the user wants multiple pages (e.g., cart, checkout), create separate HTML files.\n"
        "- After completing ALL file operations, output exactly `DONE:` followed by a brief summary.\n"
        "- Do NOT stop until you've created at least one meaningful file.\n"
        "- JSON arguments must be valid. Escape quotes properly."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Build: {description}"},
    ]

    max_turns = 10
    progress_msg = await ctx.send("🤖 Building...")
    async with ctx.typing():
        try:
            for turn in range(max_turns):
                resp = await client.chat.completions.create(
                    model=model, messages=messages, tools=tools, tool_choice="auto", temperature=0.7
                )
                msg = resp.choices[0].message
                messages.append(msg)
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": "Invalid JSON, correct and retry."})
                            continue
                        func = tc.function.name
                        if func == "create_file":
                            filename = args["filename"]
                            content = args["content"]
                            (proj_path / filename).parent.mkdir(parents=True, exist_ok=True)
                            async with aiofiles.open(proj_path / filename, "w") as f:
                                await f.write(content)
                            result = f"Created {filename}"
                            await progress_msg.edit(content=f"📁 Created `{filename}`...")
                        elif func == "read_file":
                            fpath = proj_path / args["filename"]
                            if fpath.exists():
                                async with aiofiles.open(fpath, "r") as f:
                                    result = await f.read()
                            else:
                                result = "File not found."
                        elif func == "list_files":
                            files = [str(p.relative_to(proj_path)) for p in proj_path.rglob("*") if p.is_file()]
                            result = "\n".join(files) or "No files."
                        elif func == "delete_file":
                            fpath = proj_path / args["filename"]
                            if fpath.exists():
                                fpath.unlink()
                                result = f"Deleted {args['filename']}"
                            else:
                                result = "File not found."
                        else:
                            result = "Unknown function."
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                else:
                    content = msg.content or ""
                    if "DONE:" in content:
                        break
                    else:
                        messages.append({"role": "user", "content": "Are you done? If so, say DONE:. Otherwise continue."})
            # Final summary
            final_files = [str(p.relative_to(proj_path)) for p in proj_path.rglob("*") if p.is_file()]
            render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
            file_list = "\n".join(f"• `{f}`" for f in final_files)
            await progress_msg.edit(content=f"🌐 {render_url}/{proj_name}/\n**Files:**\n{file_list}\n✅ Build complete.")
            workspace_meta[proj_name] = time.time()
            save_workspace_meta()
            asyncio.create_task(schedule_project_deletion(proj_name))
        except Exception as e:
            await progress_msg.edit(content=f"❌ Error: {str(e)}")

# -------------------------------------------------------------------
# !edit – modify existing file with AI
# -------------------------------------------------------------------
@bot.command(name="edit")
async def edit_file(ctx, filename: str, *, instruction: str):
    uid = str(ctx.author.id)
    proj = user_data.get(uid, {}).get("current_project")
    if not proj: return await ctx.send("No active project.")
    file_path = WORKSPACE_DIR / proj / filename
    if not file_path.exists(): return await ctx.send("File not found.")
    async with aiofiles.open(file_path, "r") as f:
        original = await f.read()
    client = get_user_client(uid)
    if not client: return await ctx.send("Set up provider first.")
    model = get_user_model(uid)
    async with ctx.typing():
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a code editor. Modify the file as instructed and return ONLY the new content."},
                    {"role": "user", "content": f"File: {filename}\nContent:\n{original}\n\nInstruction: {instruction}"}
                ],
                temperature=0.2,
            )
            new_content = resp.choices[0].message.content
            async with aiofiles.open(file_path, "w") as f:
                await f.write(new_content)
            await ctx.send(f"✅ `{filename}` updated. Use `!viewfile {filename}` to see changes.")
        except Exception as e:
            await ctx.send(f"❌ Error: {e}")

# -------------------------------------------------------------------
# !attach – build from image (vision model)
# -------------------------------------------------------------------
@bot.command(name="attach")
async def attach_image(ctx, *, description: str = ""):
    if not ctx.message.attachments:
        return await ctx.send("Please attach an image with `!attach`.")
    uid = str(ctx.author.id)
    if uid not in user_data: return await ctx.send("Set up provider first.")
    client = get_user_client(uid)
    if not client: return await ctx.send("Provider error.")
    provider = user_data[uid]["provider"]
    model = get_user_model(uid)
    # Try to switch to a vision model if available
    if "vision" not in model.lower() and "gemini" not in model.lower() and "gpt-4" not in model.lower():
        models = await fetch_models(provider, user_data[uid]["api_key"])
        vision_candidates = [m for m in models if ("vision" in m.lower() or "gemini" in m.lower() or "gpt-4" in m.lower())]
        if vision_candidates:
            model = vision_candidates[0]
            await ctx.send(f"ℹ️ Switching to vision model `{model}` for this request.")
        else:
            return await ctx.send("❌ Your provider has no vision model. Use `!models` to pick one.")
    attachment = ctx.message.attachments[0]
    image_bytes = await attachment.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"Build a website based on this image. {description}"},
            {"type": "image_url", "image_url": {"url": f"data:{attachment.content_type};base64,{image_b64}"}}
        ]}
    ]
    async with ctx.typing():
        try:
            resp = await client.chat.completions.create(model=model, messages=messages, temperature=0.7)
            ai_plan = resp.choices[0].message.content
            # Reuse the make command with the AI's generated description
            await ctx.invoke(bot.get_command("make"), description=ai_plan)
        except Exception as e:
            await ctx.send(f"❌ Error: {e}")

# -------------------------------------------------------------------
# !draw – image generation (if provider supports it)
# -------------------------------------------------------------------
@bot.command(name="draw")
async def generate_image(ctx, *, prompt: str):
    uid = str(ctx.author.id)
    client = get_user_client(uid)
    if not client: return await ctx.send("Set up provider first.")
    try:
        resp = await client.images.generate(model="dall-e-2", prompt=prompt, n=1, size="512x512")
        image_url = resp.data[0].url
        await ctx.send(f"🖼️ Generated: {image_url}")
    except Exception:
        await ctx.send("❌ Image generation not supported by your provider. Try using a provider that offers it (e.g., OpenRouter with FLUX).")

# -------------------------------------------------------------------
# !search – DuckDuckGo AI-filtered search
# -------------------------------------------------------------------
@bot.command(name="search")
async def search_web(ctx, *, query: str):
    uid = str(ctx.author.id)
    client = get_user_client(uid)
    if not client: return await ctx.send("Set up provider first.")
    url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return await ctx.send("Search failed.")
                data = await resp.json()
                abstract = data.get("AbstractText", "")
                related = data.get("RelatedTopics", [])
                snippets = []
                if abstract:
                    snippets.append(abstract)
                for topic in related[:5]:
                    if isinstance(topic, dict) and "Text" in topic:
                        snippets.append(topic["Text"])
                search_text = "\n".join(snippets)
                if not search_text:
                    return await ctx.send("No results.")
        except Exception as e:
            return await ctx.send(f"Search error: {e}")
    model = get_user_model(uid)
    messages = [
        {"role": "system", "content": "You are a search assistant. Use the provided search results to answer the user's question concisely, filtering out irrelevant info."},
        {"role": "user", "content": f"Question: {query}\n\nSearch results:\n{search_text}"}
    ]
    async with ctx.typing():
        try:
            resp = await client.chat.completions.create(model=model, messages=messages, temperature=0.5, max_tokens=500)
            await ctx.send(resp.choices[0].message.content[:2000])
        except Exception as e:
            await ctx.send(f"❌ AI error: {e}")

# -------------------------------------------------------------------
# Multiple commands in one message (e.g., !newproject && !make tic tac toe)
# -------------------------------------------------------------------
@bot.event
async def on_message(message):
    if message.author.bot:
        return
    if message.content.startswith("!") and " && " in message.content:
        parts = [p.strip() for p in message.content.split(" && ")]
        for part in parts:
            if part:
                # Create a copy of the message with modified content
                # We can't easily create a new message, but we can call process_commands directly
                # by modifying the message's content temporarily (safe)
                original = message.content
                message.content = part
                await bot.process_commands(message)
                message.content = original
        return
    await bot.process_commands(message)

# -------------------------------------------------------------------
# Secret admin cleanup
# -------------------------------------------------------------------
@bot.command(name="devcleanup")
async def dev_cleanup(ctx):
    try:
        await ctx.author.send("🔐 Enter admin password:")
    except discord.Forbidden:
        return await ctx.send("I cannot DM you.")
    def check(m): return m.author == ctx.author and isinstance(m.channel, discord.DMChannel)
    try:
        msg = await bot.wait_for("message", timeout=30.0, check=check)
        if msg.content == ADMIN_PASSWORD:
            for folder in list(workspace_meta.keys()):
                await delete_project(folder)
            for uid in user_data:
                user_data[uid]["current_project"] = None
            save_user_data()
            await ctx.author.send("🗑️ All projects wiped.")
        else:
            await ctx.author.send("❌ Wrong password.")
    except asyncio.TimeoutError:
        await ctx.author.send("⌛ Timed out.")

# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("❌ DISCORD_TOKEN missing.")
        exit(1)
    bot.run(DISCORD_TOKEN)
