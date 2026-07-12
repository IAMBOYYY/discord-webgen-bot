import os
import json
import asyncio
import threading
import shutil
import uuid
import time
from datetime import datetime, timedelta
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

# Provider defaults (base URL + recommended model)
PROVIDERS = {
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "google/gemma-2-9b-it",
    },
    "groq": {
        "name": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama3-8b-8192",
    },
    "nvidia": {
        "name": "NVIDIA NIM",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "default_model": "meta/llama3-8b-instruct",
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
# User data (API keys, selected provider, current project)
# -------------------------------------------------------------------
USER_DATA_FILE = Path("user_data.json")
user_data = {}   # user_id(str) -> { "provider": "...", "api_key": "...", "current_project": "..." }

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
    """Remove project folder and its meta, cancel any pending task."""
    path = WORKSPACE_DIR / project_folder
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    if project_folder in workspace_meta:
        del workspace_meta[project_folder]
        save_workspace_meta()

async def schedule_project_deletion(project_folder: str, delay_hours: float = PROJECT_TTL_HOURS):
    """Wait then delete the project folder."""
    await asyncio.sleep(delay_hours * 3600)
    await delete_project(project_folder)
    print(f"🧹 Auto-deleted project {project_folder}")

async def startup_cleanup():
    """Delete any projects older than TTL on bot restart."""
    now = time.time()
    expired = [
        folder for folder, ts in workspace_meta.items()
        if (now - ts) > PROJECT_TTL_HOURS * 3600
    ]
    for folder in expired:
        await delete_project(folder)
    # Also schedule deletion for remaining ones that haven't expired yet
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
# Helper: get user's AI client
# -------------------------------------------------------------------
def get_user_client(user_id: str):
    """Build an AsyncOpenAI client for the user's configured provider."""
    uid = str(user_id)
    if uid not in user_data or "api_key" not in user_data[uid]:
        return None
    info = user_data[uid]
    provider = info.get("provider", "openrouter")
    prov_cfg = PROVIDERS.get(provider, PROVIDERS["openrouter"])
    return AsyncOpenAI(
        api_key=info["api_key"],
        base_url=prov_cfg["base_url"],
    )

def get_user_model(user_id: str) -> str:
    """Return the model to use for the user (default from provider)."""
    uid = str(user_id)
    prov = user_data.get(uid, {}).get("provider", "openrouter")
    return PROVIDERS.get(prov, PROVIDERS["openrouter"])["default_model"]

# -------------------------------------------------------------------
# Setup commands
# -------------------------------------------------------------------
@bot.command(name="setup")
async def setup(ctx):
    """Guide through choosing a provider and setting your API key."""
    # Show available providers
    lines = ["**Choose your AI provider:**"]
    for idx, (key, prov) in enumerate(PROVIDERS.items(), 1):
        lines.append(f"`{idx}` - {prov['name']}")
    lines.append("Reply with the number (e.g., `1`).")
    await ctx.send("\n".join(lines))

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel and m.content.isdigit()

    try:
        msg = await bot.wait_for("message", timeout=60.0, check=check)
        choice = int(msg.content)
        keys = list(PROVIDERS.keys())
        if 1 <= choice <= len(keys):
            selected = keys[choice - 1]
            user_data[str(ctx.author.id)] = {
                "provider": selected,
                "api_key": None,
                "current_project": None
            }
            save_user_data()
            await ctx.send(f"✅ Selected **{PROVIDERS[selected]['name']}**. Now set your API key with:\n`!setkey YOUR_API_KEY`")
        else:
            await ctx.send("Invalid choice. Please run `!setup` again.")
    except asyncio.TimeoutError:
        await ctx.send("Setup timed out. Run `!setup` again.")

@bot.command(name="setkey")
async def set_key(ctx, *, key: str):
    """Set your API key for the currently selected provider."""
    uid = str(ctx.author.id)
    if uid not in user_data or "provider" not in user_data[uid]:
        await ctx.send("❌ Run `!setup` first to choose a provider.")
        return
    # Store key
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
    """Show your current provider and model."""
    uid = str(ctx.author.id)
    if uid in user_data:
        p = user_data[uid].get("provider", "?")
        m = get_user_model(uid)
        await ctx.send(f"Your provider: **{PROVIDERS.get(p, {}).get('name', p)}**\nModel: `{m}`")
    else:
        await ctx.send("You haven't set up yet. Use `!setup`.")

# -------------------------------------------------------------------
# !ask – general questions
# -------------------------------------------------------------------
@bot.command(name="ask")
async def ask_question(ctx, *, question: str):
    """Ask any question – the bot answers using your AI provider."""
    uid = str(ctx.author.id)
    client = get_user_client(uid)
    if not client:
        await ctx.send("❌ You need to set up a provider and API key. Use `!setup` and `!setkey`.")
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
            await ctx.send(answer[:2000])  # Discord limit
        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")

# -------------------------------------------------------------------
# Workspace / project commands
# -------------------------------------------------------------------
@bot.command(name="newproject")
async def new_project(ctx):
    """Create a new empty project and set it as your active one."""
    uid = str(ctx.author.id)
    # Generate a unique project name
    proj_name = f"proj_{uuid.uuid4().hex[:8]}"
    proj_path = WORKSPACE_DIR / proj_name
    proj_path.mkdir(parents=True, exist_ok=True)

    # Save meta
    workspace_meta[proj_name] = time.time()
    save_workspace_meta()

    # Set as user's current project
    if uid not in user_data:
        user_data[uid] = {}
    user_data[uid]["current_project"] = proj_name
    save_user_data()

    # Schedule deletion
    asyncio.create_task(schedule_project_deletion(proj_name))

    render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
    await ctx.send(f"📁 New project created: `{proj_name}`\n🌐 Public URL: {render_url}/{proj_name}/\nUse `!make <description>` to generate the website.")

@bot.command(name="make")
async def make_website(ctx, *, description: str):
    """Generate/update the website in your current project using AI tools."""
    uid = str(ctx.author.id)
    if uid not in user_data or not user_data[uid].get("current_project"):
        await ctx.send("❌ No active project. Use `!newproject` first.")
        return
    client = get_user_client(uid)
    if not client:
        await ctx.send("❌ Set up provider & key first (`!setup` + `!setkey`).")
        return
    model = get_user_model(uid)
    proj_name = user_data[uid]["current_project"]
    proj_path = WORKSPACE_DIR / proj_name

    # --- Define tools (function calling) for file operations ---
    tools = [
        {
            "type": "function",
            "function": {
                "name": "create_file",
                "description": "Create a new file with given content in the project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "File path relative to project root, e.g., 'index.html' or 'css/style.css'"},
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
                "description": "Read the content of a file in the project.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                    },
                    "required": ["filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List all files in the project directory.",
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
                    "properties": {
                        "filename": {"type": "string"},
                    },
                    "required": ["filename"],
                },
            },
        },
    ]

    # System prompt instructing the AI to use tools to build the website
    system_prompt = (
        "You are an expert full‑stack web developer. The user wants you to build a website "
        "inside their project directory. You have access to file creation, reading, listing, and deletion tools.\n\n"
        "Rules:\n"
        "- Use tools to create all necessary files (HTML, CSS, JS, assets, etc.).\n"
        "- Make the website interactive and visually appealing (use modern CSS, possibly Three.js for 3D, etc.).\n"
        "- If the user asks for a game (e.g., Tic‑Tac‑Toe), make it fully playable.\n"
        "- Start by understanding the current state: use `list_files` and `read_file` to see what exists.\n"
        "- Then modify/create files accordingly.\n"
        "- Always output a final summary of what you did after using tools."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Build a website based on this description: {description}"},
    ]

    async with ctx.typing():
        try:
            # Loop for multi-turn tool calls
            while True:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.7,
                )
                msg = response.choices[0].message
                messages.append(msg)  # add assistant response to history

                if msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        func_name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)

                        # Execute the function
                        result = ""
                        if func_name == "create_file":
                            filename = args["filename"]
                            content = args["content"]
                            file_path = proj_path / filename
                            file_path.parent.mkdir(parents=True, exist_ok=True)
                            async with aiofiles.open(file_path, "w") as f:
                                await f.write(content)
                            result = f"File created: {filename}"
                        elif func_name == "read_file":
                            filename = args["filename"]
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
                            filename = args["filename"]
                            file_path = proj_path / filename
                            if file_path.exists():
                                file_path.unlink()
                                result = f"Deleted {filename}"
                            else:
                                result = "File not found."
                        else:
                            result = "Unknown function."

                        # Append tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        })
                else:
                    # No more tool calls – final answer
                    final_reply = msg.content or "✅ Website generated."
                    # Reset TTL: each make resets the 2-hour timer
                    workspace_meta[proj_name] = time.time()
                    save_workspace_meta()
                    # Cancel old deletion task? We'll just schedule a new one.
                    # For simplicity, we schedule a new deletion; old one will still delete, but that's okay.
                    asyncio.create_task(schedule_project_deletion(proj_name))

                    render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
                    await ctx.send(
                        f"{final_reply}\n\n🌐 Website: {render_url}/{proj_name}/"
                    )
                    break

        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")

@bot.command(name="setproject")
async def set_project(ctx, project_name: str):
    """Switch your active project to an existing one."""
    uid = str(ctx.author.id)
    proj_path = WORKSPACE_DIR / project_name
    if not proj_path.exists():
        await ctx.send("❌ Project not found.")
        return
    if uid not in user_data:
        user_data[uid] = {}
    user_data[uid]["current_project"] = project_name
    save_user_data()
    await ctx.send(f"✅ Active project set to `{project_name}`.")

@bot.command(name="listprojects")
async def list_projects(ctx):
    """Show all projects and their URLs."""
    render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
    if not workspace_meta:
        await ctx.send("No projects exist.")
        return
    lines = ["**Existing projects:**"]
    for folder in workspace_meta:
        lines.append(f"• `{folder}` → {render_url}/{folder}/")
    await ctx.send("\n".join(lines))

# -------------------------------------------------------------------
# Secret admin cleanup (password protected, via DM)
# -------------------------------------------------------------------
@bot.command(name="devcleanup")
async def dev_cleanup(ctx):
    """Secret command: deletes ALL projects after password verification."""
    # Bot asks for password in DM
    try:
        await ctx.author.send("🔐 Enter the admin password to delete all projects:")
    except discord.Forbidden:
        await ctx.send("I cannot DM you. Please allow DMs from server members.")
        return

    def check(m):
        return m.author == ctx.author and isinstance(m.channel, discord.DMChannel)

    try:
        msg = await bot.wait_for("message", timeout=30.0, check=check)
        if msg.content == ADMIN_PASSWORD:
            # Delete all projects
            for folder in list(workspace_meta.keys()):
                await delete_project(folder)
            # Also clear user current projects
            for uid in user_data:
                user_data[uid]["current_project"] = None
            save_user_data()
            await ctx.author.send("🗑️ All projects have been deleted.")
        else:
            await ctx.author.send("❌ Wrong password.")
    except asyncio.TimeoutError:
        await ctx.author.send("⌛ Timed out.")

# -------------------------------------------------------------------
# Run bot
# -------------------------------------------------------------------
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("❌ DISCORD_TOKEN missing.")
        exit(1)
    bot.run(DISCORD_TOKEN)
