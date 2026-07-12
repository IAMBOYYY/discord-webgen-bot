import os
import json
import asyncio
import threading
import shutil
import uuid
import time
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

# Provider defaults (base URL, default model used only as fallback)
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
}

# -------------------------------------------------------------------
# User data (API keys, selected provider, model, current project)
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
# Workspace meta (creation times for cleanup)
# -------------------------------------------------------------------
workspace_meta = {}   # project_folder_name -> timestamp (float)

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
# HTTP server (serves workspace directory)
# -------------------------------------------------------------------
class WorkspaceHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WORKSPACE_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            super().do_GET()

def start_http_server():
    server = HTTPServer(("0.0.0.0", PORT), WorkspaceHandler)
    print(f"🌐 Web server running on port {PORT}")
    threading.Thread(target=server.serve_forever, daemon=True).start()

# -------------------------------------------------------------------
# Cleanup logic
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
    print(f"🧹 Auto-deleted project {project_folder}")

async def startup_cleanup():
    now = time.time()
    expired = [
        folder for folder, ts in workspace_meta.items()
        if (now - ts) > PROJECT_TTL_HOURS * 3600
    ]
    for folder in expired:
        await delete_project(folder)
    for folder, ts in workspace_meta.items():
        remaining = PROJECT_TTL_HOURS * 3600 - (now - ts)
        if remaining > 0:
            asyncio.create_task(schedule_project_deletion(folder, remaining / 3600))
        else:
            await delete_project(folder)

# -------------------------------------------------------------------
# Discord bot
# -------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")
    start_http_server()
    await startup_cleanup()

# -------------------------------------------------------------------
# Helper: get user's AI client (uses their chosen model)
# -------------------------------------------------------------------
def get_user_client(user_id: str):
    uid = str(user_id)
    if uid not in user_data or "api_key" not in user_data[uid]:
        return None
    info = user_data[uid]
    provider = info.get("provider", "openrouter")
    prov_cfg = PROVIDERS.get(provider, PROVIDERS["openrouter"])
    model = info.get("model", prov_cfg["default_model"])
    return AsyncOpenAI(
        api_key=info["api_key"],
        base_url=prov_cfg["base_url"],
    )

def get_user_model(user_id: str) -> str:
    uid = str(user_id)
    if uid in user_data and "model" in user_data[uid]:
        return user_data[uid]["model"]
    # fallback to default
    provider = user_data.get(uid, {}).get("provider", "openrouter")
    return PROVIDERS[provider]["default_model"]

# -------------------------------------------------------------------
# Fetch available models from a provider's /models endpoint
# -------------------------------------------------------------------
async def fetch_models(provider: str, api_key: str) -> list:
    """Returns list of model IDs (strings)."""
    prov = PROVIDERS.get(provider)
    if not prov:
        return []
    url = f"{prov['base_url']}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                # Different providers may wrap the list differently
                models = data.get("data", data.get("models", []))
                return [m["id"] for m in models if "id" in m]
        except Exception:
            return []

# -------------------------------------------------------------------
# Interactive setup flow
# -------------------------------------------------------------------
@bot.command(name="setup")
async def setup(ctx):
    """Guide through choosing provider, setting API key, and picking a model."""
    uid = str(ctx.author.id)

    # Step 1: choose provider
    lines = ["**Choose your AI provider:**"]
    keys = list(PROVIDERS.keys())
    for i, key in enumerate(keys, 1):
        lines.append(f"`{i}` - {PROVIDERS[key]['name']}")
    lines.append("Reply with the number (e.g., `1`).")
    await ctx.send("\n".join(lines))

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel and m.content.isdigit()

    try:
        msg = await bot.wait_for("message", timeout=60.0, check=check)
        choice = int(msg.content)
        if choice < 1 or choice > len(keys):
            await ctx.send("Invalid choice. Run `!setup` again.")
            return
        selected_provider = keys[choice - 1]

        # Step 2: ask for API key
        await ctx.send(f"✅ Selected **{PROVIDERS[selected_provider]['name']}**. Now send me your API key (use a DM for safety).")
        msg = await bot.wait_for("message", timeout=120.0, check=lambda m: m.author == ctx.author and m.channel == ctx.channel)
        api_key = msg.content.strip()
        # Delete the message if in a server
        if ctx.guild:
            try:
                await msg.delete()
            except:
                pass

        # Step 3: fetch models
        await ctx.send("🔍 Fetching available models for your API key…")
        models = await fetch_models(selected_provider, api_key)
        if not models:
            # Fallback: use default model
            default_model = PROVIDERS[selected_provider]["default_model"]
            user_data[uid] = {
                "provider": selected_provider,
                "api_key": api_key,
                "model": default_model,
                "current_project": None,
            }
            save_user_data()
            await ctx.send(f"⚠️ Could not fetch model list. Using default model: `{default_model}`. You can change it later with `!models`.")
        else:
            # Show models (paginate if many)
            models_sorted = sorted(models)
            # Show up to 20 models, with numbers
            display_models = models_sorted[:20]
            lines = ["**Select a model:**"]
            for i, m in enumerate(display_models, 1):
                lines.append(f"`{i}` - `{m}`")
            if len(models_sorted) > 20:
                lines.append(f"... and {len(models_sorted) - 20} more. Type the exact model ID to pick one not listed.")
            lines.append("Reply with the number or model ID.")
            await ctx.send("\n".join(lines))

            def model_check(m):
                return m.author == ctx.author and m.channel == ctx.channel

            msg = await bot.wait_for("message", timeout=60.0, check=model_check)
            choice_text = msg.content.strip()

            # Determine if they typed a number or exact ID
            if choice_text.isdigit():
                idx = int(choice_text)
                if 1 <= idx <= len(display_models):
                    chosen_model = display_models[idx - 1]
                else:
                    await ctx.send("Invalid number. Setup cancelled. Run `!setup` again.")
                    return
            else:
                # Exact model ID
                chosen_model = choice_text
                if chosen_model not in models_sorted:
                    await ctx.send(f"Model `{chosen_model}` not found in available list. Using fallback default.")
                    chosen_model = PROVIDERS[selected_provider]["default_model"]

            # Save
            user_data[uid] = {
                "provider": selected_provider,
                "api_key": api_key,
                "model": chosen_model,
                "current_project": None,
            }
            save_user_data()
            await ctx.send(f"✅ Setup complete!\nProvider: **{PROVIDERS[selected_provider]['name']}**\nModel: `{chosen_model}`")

    except asyncio.TimeoutError:
        await ctx.send("⌛ Setup timed out. Run `!setup` again.")

# -------------------------------------------------------------------
# Change model later
# -------------------------------------------------------------------
@bot.command(name="models")
async def list_models(ctx):
    """List available models for your current provider and allow switching."""
    uid = str(ctx.author.id)
    if uid not in user_data or "api_key" not in user_data[uid]:
        await ctx.send("❌ You haven't set up a provider and API key. Use `!setup`.")
        return
    provider = user_data[uid].get("provider")
    api_key = user_data[uid]["api_key"]
    await ctx.send("🔍 Fetching models…")
    models = await fetch_models(provider, api_key)
    if not models:
        await ctx.send("❌ Could not retrieve models. Check your API key.")
        return
    models_sorted = sorted(models)
    display_models = models_sorted[:20]
    lines = ["**Available models (your provider):**"]
    for i, m in enumerate(display_models, 1):
        current = " (current)" if m == user_data[uid].get("model") else ""
        lines.append(f"`{i}` - `{m}`{current}")
    if len(models_sorted) > 20:
        lines.append(f"... and more. Type exact ID to switch.")
    lines.append("Reply with the number or model ID to switch, or `cancel`.")
    await ctx.send("\n".join(lines))

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel

    try:
        msg = await bot.wait_for("message", timeout=60.0, check=check)
        choice = msg.content.strip()
        if choice.lower() == "cancel":
            await ctx.send("Cancelled.")
            return
        chosen_model = None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(display_models):
                chosen_model = display_models[idx - 1]
        else:
            if choice in models_sorted:
                chosen_model = choice
        if chosen_model:
            user_data[uid]["model"] = chosen_model
            save_user_data()
            await ctx.send(f"✅ Model switched to `{chosen_model}`.")
        else:
            await ctx.send("Invalid selection.")
    except asyncio.TimeoutError:
        await ctx.send("Timed out.")

# -------------------------------------------------------------------
# Set API key separately (if needed)
# -------------------------------------------------------------------
@bot.command(name="setkey")
async def set_key(ctx, *, key: str):
    """Set your API key for the currently selected provider."""
    uid = str(ctx.author.id)
    if uid not in user_data or "provider" not in user_data[uid]:
        await ctx.send("❌ Run `!setup` first to choose a provider.")
        return
    user_data[uid]["api_key"] = key.strip()
    save_user_data()
    if ctx.guild:
        try:
            await ctx.message.delete()
        except:
            pass
        await ctx.send("✅ API key saved. For safety, use DMs next time.", delete_after=10)
    else:
        await ctx.send("✅ API key saved securely.")

@bot.command(name="provider")
async def show_provider(ctx):
    """Show your current provider, model, and project."""
    uid = str(ctx.author.id)
    if uid in user_data:
        p = user_data[uid].get("provider", "?")
        m = get_user_model(uid)
        proj = user_data[uid].get("current_project", "none")
        await ctx.send(f"Provider: **{PROVIDERS.get(p, {}).get('name', p)}**\nModel: `{m}`\nActive project: `{proj}`")
    else:
        await ctx.send("Not set up. Use `!setup`.")

# -------------------------------------------------------------------
# !ask – general questions
# -------------------------------------------------------------------
@bot.command(name="ask")
async def ask_question(ctx, *, question: str):
    """Ask any question – the bot answers using your AI provider."""
    uid = str(ctx.author.id)
    client = get_user_client(uid)
    if not client:
        await ctx.send("❌ Set up provider & key first (`!setup`).")
        return
    model = get_user_model(uid)
    async with ctx.typing():
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=0.7,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content
            await ctx.send(answer[:2000])
        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")

# -------------------------------------------------------------------
# Workspace / project commands
# -------------------------------------------------------------------
@bot.command(name="newproject")
async def new_project(ctx):
    """Create a new empty project and set it as your active one."""
    uid = str(ctx.author.id)
    proj_name = f"proj_{uuid.uuid4().hex[:8]}"
    proj_path = WORKSPACE_DIR / proj_name
    proj_path.mkdir(parents=True, exist_ok=True)
    workspace_meta[proj_name] = time.time()
    save_workspace_meta()
    if uid not in user_data:
        user_data[uid] = {}
    user_data[uid]["current_project"] = proj_name
    save_user_data()
    asyncio.create_task(schedule_project_deletion(proj_name))
    render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
    await ctx.send(f"📁 New project: `{proj_name}`\n🌐 {render_url}/{proj_name}/\nUse `!make <description>` to build.")

@bot.command(name="make")
async def make_website(ctx, *, description: str):
    """Generate/update the website in your current project using AI tools."""
    uid = str(ctx.author.id)
    if uid not in user_data or not user_data[uid].get("current_project"):
        await ctx.send("❌ No active project. Use `!newproject`.")
        return
    client = get_user_client(uid)
    if not client:
        await ctx.send("❌ Set up provider & key first (`!setup`).")
        return
    model = get_user_model(uid)
    proj_name = user_data[uid]["current_project"]
    proj_path = WORKSPACE_DIR / proj_name

    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_file",
                "description": "Create a new file with given content in the project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "Relative path, e.g., 'index.html'"},
                        "content": {"type": "string", "description": "Full file content as plain text"},
                    },
                    "required": ["filename", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the content of a file in the project.",
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
                "description": "List all files in the project.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delete_file",
                "description": "Delete a file from the project.",
                "parameters": {
                    "type": "object",
                    "properties": {"filename": {"type": "string"}},
                    "required": ["filename"],
                },
            },
        },
    ]

    system_prompt = (
        "You are an expert full-stack web developer. Build/modify the website the user describes using the provided tools.\n\n"
        "Critical JSON rules:\n"
        "- When using tools, all arguments must be valid JSON (RFC 8259).\n"
        "- Do NOT escape backslashes or quotes incorrectly. Use standard JSON escaping.\n"
        "- For strings with literal backslashes (like CSS or paths), just include them normally; the JSON parser will handle it.\n"
        "- Do NOT add extra backslashes before quotes inside string values.\n\n"
        "Other rules:\n"
        "- Use tools to create/read/delete files as needed.\n"
        "- Make the website interactive and visually appealing (CSS, JS, Three.js, etc.).\n"
        "- For games, make them fully playable.\n"
        "- Always list files first if you need to understand the current project.\n"
        "- Output a final summary when done."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Build a website: {description}"},
    ]

    async with ctx.typing():
        try:
            while True:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.7,
                )
                msg = response.choices[0].message
                messages.append(msg)

                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        func_name = tool_call.function.name
                        raw_args = tool_call.function.arguments
                        # Attempt to parse JSON, handle errors gracefully
                        try:
                            args = json.loads(raw_args)
                        except json.JSONDecodeError as e:
                            # Tell the AI the JSON was invalid and ask it to retry
                            error_msg = f"Tool call failed: invalid JSON arguments. Error: {e}. Please correct the JSON and try again. Raw arguments: {raw_args}"
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": error_msg,
                            })
                            continue  # skip execution, let AI retry

                        # Execute the tool
                        result = ""
                        if func_name == "create_file":
                            filename = args.get("filename", "untitled")
                            content = args.get("content", "")
                            file_path = proj_path / filename
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            async with aiofiles.open(file_path, "w") as f:
                                await f.write(content)
                            result = f"File created: {filename}"
                        elif func_name == "read_file":
                            filename = args.get("filename", "")
                            file_path = proj_path / filename
                            if file_path.exists():
                                async with aiofiles.open(file_path, "r") as f:
                                    content = await f.read()
                                result = content
                            else:
                                result = "File not found."
                        elif func_name == "list_files":
                            files = []
                            for p in proj_path.rglob("*"):
                                if p.is_file():
                                    files.append(str(p.relative_to(proj_path)))
                            result = "\n".join(files) if files else "No files yet."
                        elif func_name == "delete_file":
                            filename = args.get("filename", "")
                            file_path = proj_path / filename
                            if file_path.exists():
                                file_path.unlink()
                                result = f"Deleted {filename}"
                            else:
                                result = "File not found."
                        else:
                            result = "Unknown function."

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        })
                else:
                    # No more tool calls – done
                    final_reply = msg.content or "✅ Website updated."
                    workspace_meta[proj_name] = time.time()
                    save_workspace_meta()
                    asyncio.create_task(schedule_project_deletion(proj_name))
                    render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
                    await ctx.send(f"{final_reply}\n\n🌐 {render_url}/{proj_name}/")
                    break
        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")

@bot.command(name="setproject")
async def set_project(ctx, project_name: str):
    """Switch your active project to an existing one."""
    uid = str(ctx.author.id)
    if not (WORKSPACE_DIR / project_name).exists():
        await ctx.send("❌ Project not found.")
        return
    if uid not in user_data:
        user_data[uid] = {}
    user_data[uid]["current_project"] = project_name
    save_user_data()
    await ctx.send(f"✅ Active project: `{project_name}`")

@bot.command(name="listprojects")
async def list_projects(ctx):
    """Show all projects and their URLs."""
    render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
    if not workspace_meta:
        await ctx.send("No projects yet.")
        return
    lines = ["**Projects:**"]
    for folder in workspace_meta:
        lines.append(f"• `{folder}` → {render_url}/{folder}/")
    await ctx.send("\n".join(lines))

# -------------------------------------------------------------------
# Secret admin cleanup
# -------------------------------------------------------------------
@bot.command(name="devcleanup")
async def dev_cleanup(ctx):
    """Secret command: deletes ALL projects after password verification."""
    try:
        await ctx.author.send("🔐 Enter admin password:")
    except discord.Forbidden:
        await ctx.send("I cannot DM you. Enable DMs from server members.")
        return
    def check(m):
        return m.author == ctx.author and isinstance(m.channel, discord.DMChannel)
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
        print("❌ DISCORD_TOKEN not set.")
        exit(1)
    bot.run(DISCORD_TOKEN)
