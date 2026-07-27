import os
import json
import asyncio
import threading
import shutil
import uuid
import time
import base64
import io
import random
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO

import discord
from discord.ext import commands, tasks
from openai import AsyncOpenAI, RateLimitError, APIStatusError
import aiohttp
import aiofiles
from PIL import Image, ImageDraw, ImageFont

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

# Provider definitions (unchanged)
PROVIDERS = {
    "openrouter": {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "default_model": "google/gemma-2-9b-it"},
    "groq": {"name": "Groq", "base_url": "https://api.groq.com/openai/v1", "default_model": "llama-3.1-8b-instant"},
    "nvidia": {"name": "NVIDIA NIM", "base_url": "https://integrate.api.nvidia.com/v1", "default_model": "meta/llama-3.1-8b-instruct"},
    "mistral": {"name": "Mistral AI", "base_url": "https://api.mistral.ai/v1", "default_model": "mistral-small-latest"},
    "cerebras": {"name": "Cerebras", "base_url": "https://api.cerebras.ai/v1", "default_model": "llama3.1-8b"},
    "vertex": {"name": "Vertex AI", "base_url": os.getenv("VERTEX_BASE_URL", ""), "default_model": "gemini-1.5-flash"},
}

IMAGE_PROVIDERS = {
    "openrouter": {"name": "OpenRouter images", "base_url": "https://openrouter.ai/api/v1", "default_model": "stabilityai/sd3-turbo", "api_type": "openai_images"},
    "pollinations": {"name": "Pollinations AI", "base_url": "https://image.pollinations.ai/prompt", "default_model": "flux", "api_type": "pollinations"},
    "replicate": {"name": "Replicate", "base_url": "https://api.replicate.com/v1", "default_model": "stability-ai/sdxl", "api_type": "replicate"},
}

# -------------------------------------------------------------------
# Persistent storage
# -------------------------------------------------------------------
USER_DATA_FILE = Path("user_data.json")
user_data = {}
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

SERVER_DATA_FILE = Path("server_data.json")
server_data = {}
def load_server_data():
    global server_data
    if SERVER_DATA_FILE.exists():
        with open(SERVER_DATA_FILE, "r") as f:
            server_data = json.load(f)
    else:
        server_data = {}
def save_server_data():
    with open(SERVER_DATA_FILE, "w") as f:
        json.dump(server_data, f, indent=2)
load_server_data()

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
# HTTP server – serves index.html for directories
# -------------------------------------------------------------------
class WorkspaceHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WORKSPACE_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
            return
        safe_path = self.translate_path(self.path)
        path_obj = Path(safe_path)
        if path_obj.is_dir():
            index_file = path_obj / "index.html"
            if index_file.exists():
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
                with open(index_file, "rb") as f:
                    self.wfile.write(f.read())
                return
        super().do_GET()

def start_http_server():
    server = HTTPServer(("0.0.0.0", PORT), WorkspaceHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

# -------------------------------------------------------------------
# Cleanup
# -------------------------------------------------------------------
async def delete_project(folder: str):
    path = WORKSPACE_DIR / folder
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    if folder in workspace_meta:
        del workspace_meta[folder]
        save_workspace_meta()

async def schedule_project_deletion(folder: str, hours: float = PROJECT_TTL_HOURS):
    await asyncio.sleep(hours * 3600)
    await delete_project(folder)

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
intents.members = True
bot = commands.Bot(command_prefix="!", intents=intents)
bot.remove_command('help')

# Roast tracking
roast_targets = {}  # key: (guild_id, channel_id, target_user_id) -> remaining_roasts

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")
    start_http_server()
    await startup_cleanup()
    # Internal keep-alive pings self every 5 minutes
    @tasks.loop(minutes=5)
    async def keep_alive():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{PORT}/health") as resp:
                    if resp.status != 200:
                        print("⚠️ Health check failed, restarting server...")
                        # Could restart server here if needed
        except Exception as e:
            print(f"⚠️ Keep-alive error: {e}")
    keep_alive.start()

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    await send_long_message(ctx, f"❌ Error: {str(error)}")

async def send_long_message(ctx, text):
    limit = 2000
    if len(text) <= limit:
        await ctx.send(text)
        return
    lines = text.split('\n')
    buffer = ""
    for line in lines:
        if len(buffer) + len(line) + 1 > limit:
            await ctx.send(buffer)
            buffer = line + "\n"
        else:
            buffer += line + "\n"
    if buffer:
        await ctx.send(buffer)

# -------------------------------------------------------------------
# Retry wrapper – handles 429, 413, and connection errors
# -------------------------------------------------------------------
async def ai_call_with_retry(coro, max_retries=5, delay=15):
    for attempt in range(max_retries):
        try:
            return await coro
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay)
        except APIStatusError as e:
            if e.status_code in (429, 413):
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(delay * 2)
            else:
                raise
        except Exception as e:
            if "Connection" in str(e) or "timeout" in str(e).lower():
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(delay)
            else:
                raise

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def get_user_client(uid: str):
    if uid not in user_data or "api_key" not in user_data[uid]:
        return None
    info = user_data[uid]
    provider = info.get("provider", "openrouter")
    prov_cfg = PROVIDERS.get(provider)
    if not prov_cfg or not prov_cfg["base_url"]:
        return None
    return AsyncOpenAI(api_key=info["api_key"], base_url=prov_cfg["base_url"])

def get_user_model(uid: str) -> str:
    if uid in user_data and "model" in user_data[uid]:
        return user_data[uid]["model"]
    provider = user_data.get(uid, {}).get("provider", "openrouter")
    return PROVIDERS[provider]["default_model"]

async def fetch_models(provider: str, api_key: str) -> list:
    prov = PROVIDERS.get(provider)
    if not prov or not prov["base_url"]:
        return []
    url = f"{prov['base_url']}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, timeout=10) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            models = data.get("data", data.get("models", []))
            return [{"id": m["id"], "capabilities": m.get("capabilities", [])} for m in models if "id" in m]

async def fetch_image_models(provider: str, api_key: str) -> list:
    if provider == "openrouter":
        prov = PROVIDERS["openrouter"]
        url = f"{prov['base_url']}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return [m["id"] for m in data.get("data", []) if "image" in m.get("capabilities", [])]
    return [IMAGE_PROVIDERS[provider]["default_model"]]

async def show_all_models(ctx, models: list):
    if not models:
        await ctx.send("No models found.")
        return
    total = len(models)
    lines = []
    for i, model in enumerate(models):
        model_id = model["id"] if isinstance(model, dict) else model
        lines.append(f"`{i+1}` - `{model_id}`")
    full = "\n".join(lines)
    if len(full) <= 2000:
        await ctx.send(f"**All {total} models:**\n{full}")
    else:
        for i in range(0, len(lines), 50):
            chunk = lines[i:i+50]
            start = i+1
            end = min(i+50, total)
            await ctx.send(f"**Models {start}–{end} of {total}:**\n" + "\n".join(chunk))
            await asyncio.sleep(0.5)

# -------------------------------------------------------------------
# Custom !help
# -------------------------------------------------------------------
@bot.command(name="help")
async def help_command(ctx):
    await ctx.send("""
**Commands**
`!setup` – Interactive provider & model setup (sends backup)
`!restore` – Restore settings from backup file (attach .json)
`!sharesetting @user` – Share your API settings with a friend
`!models` – Switch chat model
`!setkey <key>` – Update API key
`!provider` – Show current config
`!newproject` – Create new project
`!listfiles` – List project files
`!viewfile <name>` – View file content
`!setproject <name>` – Switch active project
`!listprojects` – List all projects
`!make <desc>` – Build website/code (with self‑improvement)
`!improve` – Manually improve current project
`!edit <file> <instruction>` – Edit a file with AI
`!attach [image] <desc>` – Build from image (retries on credits)
`!draw <prompt>` – Generate an image
`!deploy <github_token> <repo>` – Deploy project to GitHub Pages
`!ask <q>` – Ask anything (now with live web search, concise answers)
`!search <q>` – Web search + AI filter
`!serverdetail <description>` – Set server description (admin)
`!banwords word1, word2, ...` – Set banned words (warns users)
`!serverbackup` / `!serverrestore` – Server settings backup
`!imgsetup` / `!imgmodels` – Image generation config
`!devcleanup` – Admin wipe (password required)
`!roast @user` – Start roasting a user for the next 10 messages
`!meme top text; bottom text [attach image]` – Create a meme
**New:** Tag the bot (@WebGenBot) with a request and it will automatically pick the right command!
Multi‑command: use ` && `.
""")

# -------------------------------------------------------------------
# !setup (full) – unchanged but I'll include it for completeness
# -------------------------------------------------------------------
@bot.command(name="setup")
async def setup(ctx):
    uid = str(ctx.author.id)
    lines = ["**Choose your AI chat provider:**"]
    keys = list(PROVIDERS.keys())
    for i, key in enumerate(keys, 1):
        lines.append(f"`{i}` - {PROVIDERS[key]['name']}")
    await ctx.send("\n".join(lines) + "\nReply with the number.")

    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel and m.content.isdigit()
    try:
        msg = await bot.wait_for("message", timeout=180.0, check=check)
        choice = int(msg.content)
        if choice < 1 or choice > len(keys):
            return await ctx.send("Invalid choice.")
        provider = keys[choice-1]
        prov_cfg = PROVIDERS[provider]

        if provider == "vertex":
            await ctx.send("Enter Vertex AI base URL:")
            url_msg = await bot.wait_for("message", timeout=180.0, check=lambda m: m.author == ctx.author and m.channel == ctx.channel)
            PROVIDERS["vertex"]["base_url"] = url_msg.content.strip()
            prov_cfg["base_url"] = url_msg.content.strip()

        await ctx.send(f"✅ Selected **{prov_cfg['name']}**. Send your API key (DM for safety). 5 min timeout.")
        msg = await bot.wait_for("message", timeout=300.0, check=lambda m: m.author == ctx.author and m.channel == ctx.channel)
        api_key = msg.content.strip()
        if ctx.guild:
            try: await msg.delete()
            except: pass

        await ctx.send("🔍 Fetching chat models…")
        models = await fetch_models(provider, api_key)
        if not models:
            default_model = prov_cfg["default_model"]
            user_data[uid] = {"provider": provider, "api_key": api_key, "model": default_model, "current_project": None}
            save_user_data()
            return await ctx.send(f"⚠️ Could not fetch models. Using default: `{default_model}`. Change with `!models`.")

        models_sorted = sorted(models, key=lambda x: x["id"] if isinstance(x, dict) else x)
        await show_all_models(ctx, models_sorted)
        await ctx.send("Reply with number or exact model ID. 5 min timeout.")

        def model_check(m):
            return m.author == ctx.author and m.channel == ctx.channel
        msg = await bot.wait_for("message", timeout=300.0, check=model_check)
        choice_text = msg.content.strip()
        chosen_model = None
        model_ids = [m["id"] if isinstance(m, dict) else m for m in models_sorted]
        if choice_text.isdigit():
            idx = int(choice_text)
            if 1 <= idx <= len(model_ids):
                chosen_model = model_ids[idx-1]
            else:
                return await ctx.send("Invalid number. Setup cancelled.")
        else:
            if choice_text in model_ids:
                chosen_model = choice_text
            else:
                return await ctx.send("Model ID not found. Setup cancelled.")

        user_data[uid] = {"provider": provider, "api_key": api_key, "model": chosen_model, "current_project": None}
        save_user_data()
        await ctx.send(f"✅ Chat setup complete!\nProvider: **{prov_cfg['name']}**\nModel: `{chosen_model}`")

        # Image setup
        await ctx.send("Do you want to set up **image generation**? (yes/no)")
        def yes_no_check(m):
            return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() in ("yes", "no")
        try:
            img_choice = await bot.wait_for("message", timeout=180.0, check=yes_no_check)
        except asyncio.TimeoutError:
            return await ctx.send("Skipping image setup. Use `!imgsetup` later.")
        if img_choice.content.lower() == "no":
            return await ctx.send("Image generation not configured. Use `!imgsetup` later.")

        img_lines = ["**Choose image generation provider:**"]
        img_keys = list(IMAGE_PROVIDERS.keys())
        for i, key in enumerate(img_keys, 1):
            img_lines.append(f"`{i}` - {IMAGE_PROVIDERS[key]['name']}")
        await ctx.send("\n".join(img_lines) + "\nReply with number.")
        msg = await bot.wait_for("message", timeout=180.0, check=check)
        img_choice_num = int(msg.content)
        if img_choice_num < 1 or img_choice_num > len(img_keys):
            return await ctx.send("Invalid choice. Image setup cancelled.")
        img_provider = img_keys[img_choice_num-1]
        img_cfg = IMAGE_PROVIDERS[img_provider]

        await ctx.send(f"✅ Selected **{img_cfg['name']}**. Enter API key (or `skip` to use your OpenRouter key if available).")
        key_msg = await bot.wait_for("message", timeout=180.0, check=lambda m: m.author == ctx.author and m.channel == ctx.channel)
        img_api_key = key_msg.content.strip()
        if img_api_key.lower() == "skip":
            if img_provider == "openrouter" and "api_key" in user_data.get(uid, {}):
                img_api_key = user_data[uid]["api_key"]
            else:
                return await ctx.send("No existing key available. Image setup cancelled.")

        await ctx.send("🔍 Fetching image models…")
        img_models = await fetch_image_models(img_provider, img_api_key)
        if not img_models:
            default_img = img_cfg["default_model"]
            user_data[uid]["image_provider"] = img_provider
            user_data[uid]["image_api_key"] = img_api_key
            user_data[uid]["image_model"] = default_img
            save_user_data()
            return await ctx.send(f"⚠️ Using default model: `{default_img}`. Change with `!imgmodels`.")

        await show_all_models(ctx, img_models)
        await ctx.send("Reply with number or exact model ID. 5 min timeout.")
        msg = await bot.wait_for("message", timeout=300.0, check=model_check)
        choice_text = msg.content.strip()
        chosen_img_model = None
        if choice_text.isdigit():
            idx = int(choice_text)
            if 1 <= idx <= len(img_models):
                chosen_img_model = img_models[idx-1]
            else:
                return await ctx.send("Invalid number. Image setup cancelled.")
        else:
            if choice_text in img_models:
                chosen_img_model = choice_text
            else:
                return await ctx.send("Model ID not found. Image setup cancelled.")

        user_data[uid]["image_provider"] = img_provider
        user_data[uid]["image_api_key"] = img_api_key
        user_data[uid]["image_model"] = chosen_img_model
        save_user_data()
        await ctx.send(f"✅ Image setup complete!\nProvider: **{img_cfg['name']}**\nModel: `{chosen_img_model}`")

        # Send backup file
        data = user_data[uid]
        json_str = json.dumps(data, indent=2)
        file = discord.File(io.BytesIO(json_str.encode()), filename="user_backup.json")
        await ctx.send("📁 Your backup file. If the bot restarts, use `!restore` with this file to get everything back.", file=file)

    except asyncio.TimeoutError:
        await ctx.send("⌛ Setup timed out.")

# -------------------------------------------------------------------
# !sharesetting – share your API settings with another user
# -------------------------------------------------------------------
@bot.command(name="sharesetting")
async def share_setting(ctx, member: discord.Member):
    uid = str(ctx.author.id)
    target_uid = str(member.id)
    if uid not in user_data:
        return await ctx.send("You haven't set up your own settings yet. Use `!setup` first.")
    # Copy settings to target user
    user_data[target_uid] = user_data[uid].copy()
    save_user_data()
    await ctx.send(f"✅ Your settings have been shared with {member.mention}. They can now use the bot without setup.")

# -------------------------------------------------------------------
# !restore, !models, !setkey, !provider, !imgsetup, !imgmodels (as before)
# -------------------------------------------------------------------
@bot.command(name="restore")
async def restore_settings(ctx):
    if not ctx.message.attachments:
        return await ctx.send("Please attach the backup JSON file.")
    attachment = ctx.message.attachments[0]
    if not attachment.filename.endswith('.json'):
        return await ctx.send("Must be a .json file.")
    try:
        data = await attachment.read()
        config = json.loads(data.decode())
        uid = str(ctx.author.id)
        user_data[uid] = config
        save_user_data()
        await ctx.send("✅ Settings restored!")
    except Exception as e:
        await ctx.send(f"❌ Failed: {e}")

@bot.command(name="models")
async def list_models(ctx):
    uid = str(ctx.author.id)
    if uid not in user_data or "api_key" not in user_data[uid]:
        return await ctx.send("Run `!setup` first.")
    provider = user_data[uid]["provider"]
    api_key = user_data[uid]["api_key"]
    await ctx.send("🔍 Fetching chat models…")
    models = await fetch_models(provider, api_key)
    if not models:
        return await ctx.send("Could not fetch models.")
    await show_all_models(ctx, sorted(models, key=lambda x: x["id"] if isinstance(x, dict) else x))
    await ctx.send("Reply with number or model ID to switch (or `cancel`).")
    def check(m): return m.author == ctx.author and m.channel == ctx.channel
    try:
        msg = await bot.wait_for("message", timeout=300.0, check=check)
        if msg.content.lower() == "cancel": return
        chosen = None
        model_ids = [m["id"] if isinstance(m, dict) else m for m in models]
        if msg.content.isdigit():
            idx = int(msg.content)
            if 1 <= idx <= len(model_ids): chosen = model_ids[idx-1]
        elif msg.content in model_ids: chosen = msg.content
        if chosen:
            user_data[uid]["model"] = chosen
            save_user_data()
            await ctx.send(f"✅ Switched to `{chosen}`.")
        else:
            await ctx.send("Invalid selection.")
    except asyncio.TimeoutError:
        await ctx.send("Timed out.")

@bot.command(name="setkey")
async def set_key(ctx, *, key: str):
    uid = str(ctx.author.id)
    if uid not in user_data: return await ctx.send("Run `!setup` first.")
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
        p = user_data[uid].get("provider", "?")
        m = get_user_model(uid)
        proj = user_data[uid].get("current_project", "none")
        img = user_data[uid].get("image_model", "not set")
        await ctx.send(f"Chat: **{PROVIDERS[p]['name']}** / `{m}`\nImage: `{img}`\nProject: `{proj}`")
    else:
        await ctx.send("Not set up.")

@bot.command(name="imgsetup")
async def img_setup(ctx):
    uid = str(ctx.author.id)
    if uid not in user_data or "api_key" not in user_data[uid]: return await ctx.send("Run `!setup` first.")
    await ctx.send("Now configure image generation.")
    img_keys = list(IMAGE_PROVIDERS.keys())
    lines = ["**Choose image provider:**"] + [f"`{i}` - {IMAGE_PROVIDERS[k]['name']}" for i, k in enumerate(img_keys, 1)]
    await ctx.send("\n".join(lines))
    def check(m): return m.author == ctx.author and m.channel == ctx.channel and m.content.isdigit()
    try:
        msg = await bot.wait_for("message", timeout=180.0, check=check)
        choice = int(msg.content)
        if choice < 1 or choice > len(img_keys): return await ctx.send("Invalid choice.")
        img_provider = img_keys[choice-1]
        img_cfg = IMAGE_PROVIDERS[img_provider]
        await ctx.send(f"Selected **{img_cfg['name']}**. Enter API key (or `skip` to use existing key).")
        key_msg = await bot.wait_for("message", timeout=180.0, check=lambda m: m.author == ctx.author and m.channel == ctx.channel)
        api_key = key_msg.content.strip()
        if api_key.lower() == "skip" and img_provider == "openrouter" and "api_key" in user_data[uid]:
            api_key = user_data[uid]["api_key"]
        elif api_key.lower() == "skip": return await ctx.send("No key available.")
        await ctx.send("🔍 Fetching image models…")
        models = await fetch_image_models(img_provider, api_key)
        if not models:
            default = img_cfg["default_model"]
            user_data[uid]["image_provider"] = img_provider
            user_data[uid]["image_api_key"] = api_key
            user_data[uid]["image_model"] = default
            save_user_data()
            return await ctx.send(f"⚠️ Using default: `{default}`.")
        await show_all_models(ctx, models)
        await ctx.send("Reply with number or model ID.")
        def model_check(m): return m.author == ctx.author and m.channel == ctx.channel
        msg = await bot.wait_for("message", timeout=300.0, check=model_check)
        choice_text = msg.content.strip()
        chosen = None
        if choice_text.isdigit():
            idx = int(choice_text)
            if 1 <= idx <= len(models): chosen = models[idx-1]
        elif choice_text in models: chosen = choice_text
        if not chosen: return await ctx.send("Invalid selection.")
        user_data[uid]["image_provider"] = img_provider
        user_data[uid]["image_api_key"] = api_key
        user_data[uid]["image_model"] = chosen
        save_user_data()
        await ctx.send(f"✅ Image gen ready: `{chosen}`")
    except asyncio.TimeoutError:
        await ctx.send("⌛ Timed out.")

@bot.command(name="imgmodels")
async def list_img_models(ctx):
    uid = str(ctx.author.id)
    if uid not in user_data or "image_provider" not in user_data[uid]: return await ctx.send("No image gen configured.")
    provider = user_data[uid]["image_provider"]
    api_key = user_data[uid]["image_api_key"]
    await ctx.send("🔍 Fetching image models…")
    models = await fetch_image_models(provider, api_key)
    if not models: return await ctx.send("Could not fetch models.")
    await show_all_models(ctx, models)
    await ctx.send("Reply with number or model ID to switch (or `cancel`).")
    def check(m): return m.author == ctx.author and m.channel == ctx.channel
    try:
        msg = await bot.wait_for("message", timeout=300.0, check=check)
        if msg.content.lower() == "cancel": return
        chosen = None
        if msg.content.isdigit():
            idx = int(msg.content)
            if 1 <= idx <= len(models): chosen = models[idx-1]
        elif msg.content in models: chosen = msg.content
        if chosen:
            user_data[uid]["image_model"] = chosen
            save_user_data()
            await ctx.send(f"✅ Switched to `{chosen}`.")
        else:
            await ctx.send("Invalid selection.")
    except asyncio.TimeoutError:
        await ctx.send("Timed out.")

# -------------------------------------------------------------------
# !draw (unchanged)
# -------------------------------------------------------------------
@bot.command(name="draw")
async def draw_image(ctx, *, prompt: str):
    uid = str(ctx.author.id)
    if uid not in user_data or "image_provider" not in user_data[uid]: return await ctx.send("❌ Image gen not configured.")
    provider = user_data[uid]["image_provider"]
    api_key = user_data[uid]["image_api_key"]
    model = user_data[uid].get("image_model", IMAGE_PROVIDERS[provider]["default_model"])
    cfg = IMAGE_PROVIDERS[provider]

    enhanced_prompt = prompt
    client = get_user_client(uid)
    if client:
        try:
            chat_model = get_user_model(uid)
            resp = await ai_call_with_retry(
                client.chat.completions.create(
                    model=chat_model,
                    messages=[{"role": "system", "content": "Turn the user's prompt into a detailed image generation prompt. Output only the prompt."},
                              {"role": "user", "content": prompt}],
                    temperature=0.7, max_tokens=200,
                )
            )
            enhanced_prompt = resp.choices[0].message.content.strip()
            await ctx.send(f"✨ Enhanced: {enhanced_prompt}")
        except Exception as e:
            await ctx.send(f"⚠️ Enhancement failed ({e}), using original.")

    await ctx.send("🎨 Generating...")
    async with ctx.typing():
        try:
            if provider == "openrouter":
                image_client = AsyncOpenAI(api_key=api_key, base_url=cfg["base_url"])
                resp = await ai_call_with_retry(
                    image_client.images.generate(model=model, prompt=enhanced_prompt, n=1, size="1024x1024")
                )
                await ctx.send(f"🖼️ {resp.data[0].url}")
            elif provider == "pollinations":
                url = f"{cfg['base_url']}/{enhanced_prompt}?model={model}&key={api_key}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as img_resp:
                        if img_resp.status == 200:
                            img_data = await img_resp.read()
                            await ctx.send(file=discord.File(io.BytesIO(img_data), filename="image.png"))
                        else:
                            await ctx.send(f"Pollinations error: HTTP {img_resp.status}")
            elif provider == "replicate":
                headers = {"Authorization": f"Token {api_key}"}
                async with aiohttp.ClientSession() as session:
                    payload = {"version": model, "input": {"prompt": enhanced_prompt}}
                    async with session.post(f"{cfg['base_url']}/predictions", json=payload, headers=headers) as resp:
                        if resp.status != 201: return await ctx.send("Replicate error.")
                        data = await resp.json()
                    pred_id = data["id"]
                    await asyncio.sleep(5)
                    async with session.get(f"{cfg['base_url']}/predictions/{pred_id}", headers=headers) as resp:
                        data = await resp.json()
                    if data["status"] == "succeeded":
                        await ctx.send(f"🖼️ {data['output'][0]}")
                    else:
                        await ctx.send("Still processing, check later.")
        except Exception as e:
            await ctx.send(f"❌ Image generation failed: {e}")

# -------------------------------------------------------------------
# !attach (unchanged)
# -------------------------------------------------------------------
@bot.command(name="attach")
async def attach_image(ctx, *, description: str = ""):
    if not ctx.message.attachments: return await ctx.send("Attach an image.")
    uid = str(ctx.author.id)
    if uid not in user_data: return await ctx.send("Set up provider first.")
    client = get_user_client(uid)
    if not client: return await ctx.send("Provider error.")
    provider = user_data[uid]["provider"]
    api_key = user_data[uid]["api_key"]

    models = await fetch_models(provider, api_key)
    vision_models = [m["id"] for m in models if isinstance(m, dict) and "image" in m.get("capabilities", [])]
    if not vision_models:
        if provider == "openrouter":
            vision_models = ["google/gemini-2.0-flash-001", "openai/gpt-4o", "anthropic/claude-3.5-sonnet"]
        else:
            return await ctx.send("No vision model found.")
    model = vision_models[0]
    await ctx.send(f"ℹ️ Using vision model `{model}`.")

    attachment = ctx.message.attachments[0]
    if attachment.size > 500_000: await ctx.send("⚠️ Image is large.")
    image_bytes = await attachment.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": f"Build a website based on this image. {description}"},
        {"type": "image_url", "image_url": {"url": f"data:{attachment.content_type};base64,{image_b64}"}}
    ]}]
    for attempt in range(3):
        async with ctx.typing():
            try:
                resp = await ai_call_with_retry(
                    client.chat.completions.create(model=model, messages=messages, temperature=0.7, max_tokens=256)
                )
                ai_plan = resp.choices[0].message.content
                return await ctx.invoke(bot.get_command("make"), description=ai_plan)
            except Exception as e:
                if "402" in str(e) and "credits" in str(e):
                    if attempt == 2: return await ctx.send("❌ Out of credits.")
                    await ctx.send(f"⏳ Credit limit, waiting 30s...")
                    await asyncio.sleep(30)
                else:
                    return await ctx.send(f"❌ Error: {e}")

# -------------------------------------------------------------------
# !serverdetail – now only sets description
# -------------------------------------------------------------------
@bot.command(name="serverdetail")
@commands.has_permissions(administrator=True)
async def server_detail(ctx, *, description: str):
    gid = str(ctx.guild.id)
    if gid not in server_data:
        server_data[gid] = {}
    server_data[gid]["description"] = description
    save_server_data()
    await ctx.send("✅ Server description updated.")

# -------------------------------------------------------------------
# !banwords – sets banned words (comma separated)
# -------------------------------------------------------------------
@bot.command(name="banwords")
@commands.has_permissions(administrator=True)
async def set_banwords(ctx, *, words: str):
    gid = str(ctx.guild.id)
    word_list = [w.strip().lower() for w in words.split(",") if w.strip()]
    if gid not in server_data:
        server_data[gid] = {}
    server_data[gid]["banned_words"] = word_list
    save_server_data()
    await ctx.send(f"✅ Banned words set: {', '.join(word_list)}")

# -------------------------------------------------------------------
# !serverbackup / !serverrestore
# -------------------------------------------------------------------
@bot.command(name="serverbackup")
@commands.has_permissions(administrator=True)
async def server_backup(ctx):
    gid = str(ctx.guild.id)
    if gid not in server_data: return await ctx.send("No server settings.")
    data = server_data[gid]
    json_str = json.dumps(data, indent=2)
    file = discord.File(io.BytesIO(json_str.encode()), filename="server_backup.json")
    await ctx.send("📁 Server backup.", file=file)

@bot.command(name="serverrestore")
@commands.has_permissions(administrator=True)
async def server_restore(ctx):
    if not ctx.message.attachments: return await ctx.send("Attach server_backup.json.")
    try:
        data = await ctx.message.attachments[0].read()
        config = json.loads(data.decode())
        gid = str(ctx.guild.id)
        server_data[gid] = config
        save_server_data()
        await ctx.send("✅ Server settings restored.")
    except Exception as e:
        await ctx.send(f"❌ Error: {e}")

# -------------------------------------------------------------------
# Auto-moderation – warns only, does not delete
# -------------------------------------------------------------------
@bot.event
async def on_message(message):
    if message.author.bot: return
    # Roast handling
    guild_id = message.guild.id if message.guild else None
    channel_id = message.channel.id
    author_id = message.author.id
    if guild_id:
        key = (guild_id, channel_id, author_id)
        if key in roast_targets and roast_targets[key] > 0:
            await handle_roast(message, key)
            return  # don't process further if we just roasted

    # Server banned words
    if message.guild:
        gid = str(message.guild.id)
        if gid in server_data and server_data[gid].get("banned_words"):
            banned = server_data[gid]["banned_words"]
            content_lower = message.content.lower()
            for word in banned:
                if word in content_lower:
                    try:
                        await message.reply(f"{message.author.mention} Please avoid using that word.")
                    except:
                        pass
                    break

    # Multi-command handler
    if message.content.startswith("!") and " && " in message.content:
        parts = [p.strip() for p in message.content.split(" && ")]
        for part in parts:
            if part:
                original = message.content
                message.content = part
                await bot.process_commands(message)
                message.content = original
        return

    # Mention handling: if the bot is mentioned, try to decide which command to run
    if bot.user in message.mentions:
        content = message.content.replace(f'<@{bot.user.id}>', '').strip()
        if not content:
            await message.reply("Yes? Use `!help` to see what I can do.")
            return
        # Simple intent recognition
        cmd = None
        content_lower = content.lower()
        if any(kw in content_lower for kw in ["draw", "image", "picture", "generate image"]):
            cmd = bot.get_command("draw")
        elif any(kw in content_lower for kw in ["make", "build", "create", "website", "webpage", "code", "app"]):
            cmd = bot.get_command("make")
        elif any(kw in content_lower for kw in ["search", "google", "find"]):
            cmd = bot.get_command("search")
        elif any(kw in content_lower for kw in ["ask", "question", "what", "who", "how", "why", "when", "where"]):
            cmd = bot.get_command("ask")
        elif "roast" in content_lower:
            # Could parse user mention? but need target
            await message.reply("To roast someone, use `!roast @user`")
            return
        else:
            # Default to ask
            cmd = bot.get_command("ask")

        # If cmd found, invoke it with the content as argument
        if cmd:
            # Pass the remaining text as the argument
            await ctx.invoke(cmd, content)
        return

    await bot.process_commands(message)

# -------------------------------------------------------------------
# Roast system
# -------------------------------------------------------------------
async def handle_roast(message, key):
    """Generate a roast reply for the target user."""
    uid = str(message.author.id)  # the user being roasted
    # Get the user who initiated the roast (stored elsewhere? We'll store initiator in roast_targets value as a tuple)
    # For simplicity, we'll just use the bot to generate a roast without needing the initiator.
    target_user = message.author
    # Use the first available user's API key? Better to use the roaster's key. We'll store roaster_id in the dict.
    # Modify roast_targets to store (remaining, roaster_id)
    # I'll adapt: key: (guild, channel, target) -> (roasts_left, roaster_id)
    # We'll update the !roast command to set that.

    # This function will be called after we update the data structure.
    # For now, I'll assume the data is stored as tuple (count, roaster_id).
    data = roast_targets.get(key)
    if not data:
        return
    count, roaster_id = data
    if count <= 0:
        del roast_targets[key]
        return

    # Use the roaster's API to generate a roast
    roaster_uid = str(roaster_id)
    client = get_user_client(roaster_uid)
    if not client:
        await message.reply("Roast machine broken (no API key).")
        return
    model = get_user_model(roaster_uid)
    prompt = f"Roast this Discord user in a funny, friendly, but savage way. Mention their name '{target_user.display_name}'. Keep it short, one sentence."
    try:
        resp = await ai_call_with_retry(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=60
            )
        )
        roast = resp.choices[0].message.content.strip()
        await message.reply(f"{target_user.mention} {roast}")
    except Exception as e:
        await message.reply(f"{target_user.mention} Oops, I couldn't roast you: {e}")

    # Decrement count
    new_count = count - 1
    if new_count <= 0:
        del roast_targets[key]
    else:
        roast_targets[key] = (new_count, roaster_id)

@bot.command(name="roast")
async def roast_start(ctx, member: discord.Member):
    """Start roasting the mentioned user for the next 10 messages."""
    if member == ctx.author:
        return await ctx.send("You can't roast yourself, that's sad.")
    if member.bot:
        return await ctx.send("I don't roast bots... yet.")
    key = (ctx.guild.id, ctx.channel.id, member.id)
    roast_targets[key] = (10, ctx.author.id)
    await ctx.send(f"🔥 {member.mention}, you're about to get roasted for the next 10 messages! Say something if you dare.")

# -------------------------------------------------------------------
# !meme command
# -------------------------------------------------------------------
@bot.command(name="meme")
async def meme_command(ctx, *, texts: str):
    """Create a meme: !meme top text; bottom text [attach image]"""
    if not ctx.message.attachments:
        return await ctx.send("Please attach an image to make a meme.")
    attachment = ctx.message.attachments[0]
    # Split texts by semicolon
    parts = texts.split(';')
    top_text = parts[0].strip() if len(parts) > 0 else ""
    bottom_text = parts[1].strip() if len(parts) > 1 else ""

    # Download image
    async with aiohttp.ClientSession() as session:
        async with session.get(attachment.url) as resp:
            if resp.status != 200:
                return await ctx.send("Failed to download image.")
            img_data = await resp.read()

    image = Image.open(io.BytesIO(img_data)).convert("RGB")
    draw = ImageDraw.Draw(image)
    # Use a default font, or try to load a system font
    try:
        font = ImageFont.truetype("arial.ttf", size=40)
    except:
        font = ImageFont.load_default()

    # Calculate text position (centered)
    def draw_text(draw, text, y_start, max_width):
        lines = []
        words = text.split()
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            bbox = draw.textbbox((0,0), test_line, font=font)
            if bbox[2] <= max_width:
                line = test_line
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)
        y = y_start
        for line in lines:
            bbox = draw.textbbox((0,0), line, font=font)
            w = bbox[2] - bbox[0]
            x = (max_width - w) // 2
            draw.text((x, y), line, fill="white", font=font, stroke_width=2, stroke_fill="black")
            y += bbox[3] - bbox[1] + 10
        return y

    max_width = image.width - 20
    if top_text:
        draw_text(draw, top_text, 10, max_width)
    if bottom_text:
        # Position from bottom
        bottom_y = image.height - 100
        draw_text(draw, bottom_text, bottom_y, max_width)

    # Save to BytesIO
    output = io.BytesIO()
    image.save(output, format="PNG")
    output.seek(0)
    await ctx.send(file=discord.File(output, filename="meme.png"))

# -------------------------------------------------------------------
# Welcome message system
# -------------------------------------------------------------------
@bot.command(name="welcome")
@commands.has_permissions(administrator=True)
async def welcome_set(ctx, *, message: str):
    gid = str(ctx.guild.id)
    if gid not in server_data: server_data[gid] = {}
    server_data[gid]["welcome_message"] = message
    save_server_data()
    await ctx.send("✅ Welcome message set!")

@bot.event
async def on_member_join(member):
    gid = str(member.guild.id)
    if gid in server_data and server_data[gid].get("welcome_message"):
        msg = server_data[gid]["welcome_message"].replace("{user}", member.mention).replace("{server}", member.guild.name)
        try:
            await member.send(msg)
        except discord.Forbidden:
            pass

# -------------------------------------------------------------------
# !ask – NOW WITH WEB SEARCH INTEGRATION and concise answers
# -------------------------------------------------------------------
SERVER_KEYWORDS = ["server", "guild", "member", "owner", "moderator", "admin", "channel", "role", "emojis", "boost", "level", "created", "this place", "here", "how many people", "who owns", "who is online", "online members"]
def is_server_question(q: str) -> bool:
    return any(word in q.lower() for word in SERVER_KEYWORDS)
def get_server_info(guild: discord.Guild) -> str:
    total = guild.member_count
    online_members = [m.name for m in guild.members if m.status != discord.Status.offline]
    online_str = ", ".join(online_members[:10])
    if len(online_members) > 10: online_str += f" and {len(online_members)-10} more"
    roles = ", ".join([r.name for r in guild.roles[:10]])
    owner = guild.owner.name if guild.owner else "Unknown"
    created = guild.created_at.strftime("%B %d, %Y")
    gid = str(guild.id)
    description = server_data.get(gid, {}).get("description", "No description set.")
    return (f"Server: {guild.name}\nDescription: {description}\nOwner: {owner}\nTotal members: {total}\nOnline ({len(online_members)}): {online_str}\n"
            f"Text channels: {len(guild.text_channels)}, Voice: {len(guild.voice_channels)}\nRoles: {roles}\nCreated: {created}")

async def duckduckgo_search(query: str) -> str:
    url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return ""
                data = await resp.json()
                abstract = data.get("AbstractText", "")
                related = data.get("RelatedTopics", [])
                snippets = [abstract] if abstract else []
                for topic in related[:5]:
                    if isinstance(topic, dict) and "Text" in topic:
                        snippets.append(topic["Text"])
                return "\n".join(snippets)
        except Exception:
            return ""

@bot.command(name="ask")
async def ask_question(ctx, *, question: str):
    uid = str(ctx.author.id)
    client = get_user_client(uid)
    if not client: return await ctx.send("❌ Set up provider first.")
    model = get_user_model(uid)

    # For general questions, always fetch web context
    web_context = ""
    if ctx.guild and is_server_question(question):
        web_context = ""
    else:
        web_context = await duckduckgo_search(question)
        if web_context:
            web_context = f"Web search results:\n{web_context}\n\n"

    system_content = "You are a helpful, concise assistant. Answer in 1-2 short paragraphs. Be informative but brief."
    if ctx.guild and is_server_question(question):
        system_content = f"Here is real-time server data:\n{get_server_info(ctx.guild)}\nAnswer using this data concisely."
    if web_context:
        system_content += f"\nUse the following web search results to provide an accurate, up-to-date answer:\n{web_context}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question}
    ]

    async with ctx.typing():
        try:
            resp = await ai_call_with_retry(
                client.chat.completions.create(model=model, messages=messages, temperature=0.5, max_tokens=300)
            )
            await send_long_message(ctx, resp.choices[0].message.content)
        except Exception as e:
            await ctx.send(f"❌ Error: {e}")

# -------------------------------------------------------------------
# !search – (kept for dedicated search)
# -------------------------------------------------------------------
@bot.command(name="search")
async def search_web(ctx, *, query: str):
    uid = str(ctx.author.id)
    client = get_user_client(uid)
    if not client: return await ctx.send("Set up provider first.")
    search_text = await duckduckgo_search(query)
    if not search_text: return await ctx.send("No search results.")
    model = get_user_model(uid)
    messages = [
        {"role": "system", "content": "Use the search results to answer concisely, filtering irrelevant info."},
        {"role": "user", "content": f"Question: {query}\n\nResults:\n{search_text}"}
    ]
    async with ctx.typing():
        try:
            resp = await ai_call_with_retry(
                client.chat.completions.create(model=model, messages=messages, temperature=0.5, max_tokens=300)
            )
            await send_long_message(ctx, resp.choices[0].message.content)
        except Exception as e:
            await ctx.send(f"❌ Error: {e}")

# -------------------------------------------------------------------
# !userinfo, !serverinfo, !poll, !remind, !purge, !embed, !role
# -------------------------------------------------------------------
@bot.command(name="userinfo")
async def user_info(ctx, member: discord.Member = None):
    member = member or ctx.author
    embed = discord.Embed(title="User Info", color=member.color)
    embed.set_thumbnail(url=member.display_avatar.url)
    embed.add_field(name="Name", value=member.name, inline=True)
    embed.add_field(name="ID", value=member.id, inline=True)
    embed.add_field(name="Status", value=str(member.status).title(), inline=True)
    embed.add_field(name="Top Role", value=member.top_role.mention, inline=True)
    embed.add_field(name="Joined", value=member.joined_at.strftime("%Y-%m-%d"), inline=True)
    embed.add_field(name="Created", value=member.created_at.strftime("%Y-%m-%d"), inline=True)
    await ctx.send(embed=embed)

@bot.command(name="serverinfo")
async def server_info(ctx):
    guild = ctx.guild
    embed = discord.Embed(title=guild.name, color=discord.Color.blue())
    embed.set_thumbnail(url=guild.icon.url if guild.icon else None)
    embed.add_field(name="Owner", value=guild.owner.mention, inline=True)
    embed.add_field(name="Members", value=guild.member_count, inline=True)
    embed.add_field(name="Roles", value=len(guild.roles), inline=True)
    embed.add_field(name="Channels", value=len(guild.channels), inline=True)
    embed.add_field(name="Created", value=guild.created_at.strftime("%Y-%m-%d"), inline=True)
    await ctx.send(embed=embed)

@bot.command(name="poll")
async def create_poll(ctx, question: str, *options: str):
    if len(options) < 2:
        return await ctx.send("Need at least 2 options. Usage: `!poll \"Question\" \"Opt1\" \"Opt2\" ...`")
    if len(options) > 10:
        return await ctx.send("Max 10 options.")
    embed = discord.Embed(title=question, description="\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)]))
    msg = await ctx.send(embed=embed)
    for i in range(len(options)):
        await msg.add_reaction(f"{i+1}\u20e3")

@bot.command(name="remind")
async def remind(ctx, time_str: str, *, reminder: str):
    seconds = 0
    if time_str.endswith("m"):
        seconds = int(time_str[:-1]) * 60
    elif time_str.endswith("h"):
        seconds = int(time_str[:-1]) * 3600
    elif time_str.endswith("d"):
        seconds = int(time_str[:-1]) * 86400
    else:
        return await ctx.send("Invalid time format. Use 10m, 2h, 1d.")
    await ctx.send(f"⏰ Reminder set for {time_str} from now.")
    await asyncio.sleep(seconds)
    try:
        await ctx.author.send(f"🔔 Reminder: {reminder}")
    except discord.Forbidden:
        pass

@bot.command(name="purge")
@commands.has_permissions(manage_messages=True)
async def purge(ctx, amount: int):
    await ctx.channel.purge(limit=amount + 1)
    await ctx.send(f"🧹 Cleared {amount} messages.", delete_after=5)

@bot.command(name="embed")
@commands.has_permissions(manage_messages=True)
async def embed_command(ctx, channel: discord.TextChannel, *, content: str):
    parts = content.split("|")
    title = parts[0].strip() if len(parts) > 0 else "No title"
    description = parts[1].strip() if len(parts) > 1 else "No description"
    embed = discord.Embed(title=title, description=description, color=discord.Color.green())
    await channel.send(embed=embed)
    await ctx.send("✅ Embed sent.")

@bot.command(name="role")
@commands.has_permissions(manage_roles=True)
async def role_command(ctx, member: discord.Member, *, role_name: str):
    role = discord.utils.get(ctx.guild.roles, name=role_name)
    if not role:
        return await ctx.send(f"Role `{role_name}` not found.")
    if role in member.roles:
        await member.remove_roles(role)
        await ctx.send(f"❌ Removed {role.name} from {member.display_name}.")
    else:
        await member.add_roles(role)
        await ctx.send(f"✅ Added {role.name} to {member.display_name}.")

# -------------------------------------------------------------------
# Project commands (newproject, listfiles, viewfile, setproject, listprojects)
# -------------------------------------------------------------------
@bot.command(name="newproject")
async def new_project(ctx):
    uid = str(ctx.author.id)
    proj = f"proj_{uuid.uuid4().hex[:8]}"
    (WORKSPACE_DIR / proj).mkdir(parents=True, exist_ok=True)
    workspace_meta[proj] = time.time()
    save_workspace_meta()
    if uid not in user_data: user_data[uid] = {}
    user_data[uid]["current_project"] = proj
    save_user_data()
    asyncio.create_task(schedule_project_deletion(proj))
    render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
    await ctx.send(f"📁 Project `{proj}`\n🌐 {render_url}/{proj}/")

@bot.command(name="listfiles")
async def list_files_cmd(ctx):
    uid = str(ctx.author.id)
    proj = user_data.get(uid, {}).get("current_project")
    if not proj: return await ctx.send("No active project.")
    proj_path = WORKSPACE_DIR / proj
    files = [str(p.relative_to(proj_path)) for p in proj_path.rglob("*") if p.is_file()]
    if files: await ctx.send("**Files:**\n" + "\n".join(f"• `{f}`" for f in files))
    else: await ctx.send("No files yet.")

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
async def set_project(ctx, name: str):
    uid = str(ctx.author.id)
    if not (WORKSPACE_DIR / name).exists(): return await ctx.send("Project not found.")
    user_data[uid]["current_project"] = name
    save_user_data()
    await ctx.send(f"✅ Active project: `{name}`")

@bot.command(name="listprojects")
async def list_projects(ctx):
    render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
    if not workspace_meta: return await ctx.send("No projects.")
    lines = ["**Projects:**"]
    for folder in workspace_meta: lines.append(f"• `{folder}` → {render_url}/{folder}/")
    await send_long_message(ctx, "\n".join(lines))

# -------------------------------------------------------------------
# !make (simplified, no subdirectories, with retries)
# -------------------------------------------------------------------
@bot.command(name="make")
async def make_website(ctx, *, description: str):
    uid = str(ctx.author.id)
    if uid not in user_data or not user_data[uid].get("current_project"):
        return await ctx.send("❌ No active project. Use `!newproject`.")
    client = get_user_client(uid)
    if not client: return await ctx.send("Set up provider first.")
    model = get_user_model(uid)
    proj_name = user_data[uid]["current_project"]
    proj_path = WORKSPACE_DIR / proj_name

    # Phase 1: Plan (keep it simple)
    plan_msg = await ctx.send("📝 Planning...")
    try:
        plan_resp = await ai_call_with_retry(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": "You are a planning AI. Outline a simple, minimal structure for this project. Avoid over‑engineering. Max 5‑6 files."},
                          {"role": "user", "content": description}],
                temperature=0.5, max_tokens=200
            )
        )
        plan = plan_resp.choices[0].message.content
        await plan_msg.edit(content=f"📝 Plan:\n{plan}")
    except Exception as e:
        plan = f"Build {description}"
        await plan_msg.edit(content=f"⚠️ Planning failed: {e}. Proceeding directly.")

    # Phase 2: Build with tools
    tools = [
        {"type": "function", "function": {"name": "create_file", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}, "content": {"type": "string"}}, "required": ["filename", "content"]}, "description": "Create a file"}},
        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}, "description": "Read a file"}},
        {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object", "properties": {}}, "description": "List files"}},
        {"type": "function", "function": {"name": "delete_file", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}, "description": "Delete a file"}},
    ]
    system_prompt = (
        "You are an expert web developer. Build exactly what the user asked for.\n"
        "CRITICAL RULES:\n"
        "- Do NOT create subdirectories. Place ALL files directly in the project root.\n"
        "- Keep the structure simple: one HTML file, one CSS, one JS. If multi‑page, each page is a separate HTML file in the root.\n"
        "- Use modern, responsive CSS with a nice color scheme. For images, use Unsplash URLs like `https://source.unsplash.com/random/800x600/?topic`.\n"
        "- Make games fully playable. Ensure all links between pages work.\n"
        "- After ALL file operations, output exactly `DONE:` followed by a brief summary.\n"
        "- JSON arguments must be valid. Do not halt early."
    )
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Plan:\n{plan}\n\nBuild: {description}"}]
    progress_msg = await ctx.send("🤖 Building...")
    async with ctx.typing():
        for _ in range(10):
            try:
                resp = await ai_call_with_retry(
                    client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto", temperature=0.7)
                )
            except Exception as e:
                await progress_msg.edit(content=f"❌ Build error: {e}")
                return
            msg = resp.choices[0].message
            messages.append(msg)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError as e:
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": f"Invalid JSON: {e}. Correct and retry."})
                        continue
                    func = tc.function.name
                    if func == "create_file":
                        filename = args["filename"]
                        # Remove any leading directory paths
                        filename = Path(filename).name
                        content = args["content"]
                        file_path = proj_path / filename
                        async with aiofiles.open(file_path, "w") as f:
                            await f.write(content)
                        result = f"Created {filename}"
                        await progress_msg.edit(content=f"📁 {filename}")
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
                messages.append({"role": "user", "content": "Are you done? Say DONE: if so."})

    # Phase 3: Self‑review and improve (optional, fast)
    await progress_msg.edit(content="🔍 Reviewing...")
    review_messages = [
        {"role": "system", "content": "You are a code reviewer. Check the project files and fix any obvious issues. Output `DONE:` when satisfied."},
        {"role": "user", "content": f"Files: {', '.join([str(p.relative_to(proj_path)) for p in proj_path.rglob('*') if p.is_file()])}. Review and improve."}
    ]
    for _ in range(2):
        try:
            resp = await ai_call_with_retry(
                client.chat.completions.create(model=model, messages=review_messages, tools=tools, tool_choice="auto", temperature=0.3)
            )
        except Exception:
            break
        msg = resp.choices[0].message
        review_messages.append(msg)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except:
                    review_messages.append({"role": "tool", "tool_call_id": tc.id, "content": "Invalid JSON."})
                    continue
                func = tc.function.name
                if func == "create_file":
                    filename = Path(args["filename"]).name
                    content = args["content"]
                    file_path = proj_path / filename
                    async with aiofiles.open(file_path, "w") as f:
                        await f.write(content)
                    result = f"Updated {filename}"
                elif func == "read_file" or func == "list_files":
                    pass
                else:
                    result = "Unknown function."
                review_messages.append({"role": "tool", "tool_call_id": tc.id, "content": "done"})
        else:
            if "DONE:" in (msg.content or ""):
                break

    # Final output
    final_files = [str(p.relative_to(proj_path)) for p in proj_path.rglob("*") if p.is_file()]
    render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
    summary = f"🌐 {render_url}/{proj_name}/\n**Files:**\n" + "\n".join(f"• `{f}`" for f in final_files) + "\n✅ Build complete."
    await progress_msg.edit(content=summary)
    workspace_meta[proj_name] = time.time()
    save_workspace_meta()
    asyncio.create_task(schedule_project_deletion(proj_name))

# -------------------------------------------------------------------
# !improve, !edit, !deploy (unchanged)
# -------------------------------------------------------------------
@bot.command(name="improve")
async def improve_project(ctx):
    uid = str(ctx.author.id)
    proj = user_data.get(uid, {}).get("current_project")
    if not proj: return await ctx.send("No active project.")
    client = get_user_client(uid)
    if not client: return await ctx.send("Set up provider first.")
    model = get_user_model(uid)
    proj_path = WORKSPACE_DIR / proj
    files = [str(p.relative_to(proj_path)) for p in proj_path.rglob("*") if p.is_file()]
    if not files: return await ctx.send("No files to improve.")
    await ctx.send("🔍 AI improving project...")
    tools = [
        {"type": "function", "function": {"name": "create_file", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}, "content": {"type": "string"}}, "required": ["filename", "content"]}, "description": "Create a file"}},
        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}, "description": "Read a file"}},
        {"type": "function", "function": {"name": "list_files", "parameters": {"type": "object", "properties": {}}, "description": "List files"}},
    ]
    review_messages = [
        {"role": "system", "content": "You are a code reviewer. Examine the project files and suggest improvements. Use tools to implement fixes. Output `DONE:` when satisfied."},
        {"role": "user", "content": f"Files: {', '.join(files)}. Improve the project."}
    ]
    for _ in range(3):
        try:
            resp = await ai_call_with_retry(
                client.chat.completions.create(model=model, messages=review_messages, tools=tools, tool_choice="auto", temperature=0.3)
            )
        except Exception:
            break
        msg = resp.choices[0].message
        review_messages.append(msg)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except:
                    continue
                func = tc.function.name
                if func == "create_file":
                    filename = Path(args["filename"]).name
                    content = args["content"]
                    (proj_path / filename).parent.mkdir(parents=True, exist_ok=True)
                    async with aiofiles.open(proj_path / filename, "w") as f:
                        await f.write(content)
                elif func == "read_file" or func == "list_files":
                    pass
                review_messages.append({"role": "tool", "tool_call_id": tc.id, "content": "done"})
        else:
            if "DONE:" in (msg.content or ""):
                break
    await ctx.send("✅ Improvement pass complete.")

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
            resp = await ai_call_with_retry(
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": "Edit the file as instructed. Return ONLY the new content."},
                              {"role": "user", "content": f"File: {filename}\nContent:\n{original}\n\nInstruction: {instruction}"}],
                    temperature=0.2,
                )
            )
            new_content = resp.choices[0].message.content
            async with aiofiles.open(file_path, "w") as f:
                await f.write(new_content)
            await ctx.send(f"✅ `{filename}` updated.")
        except Exception as e:
            await ctx.send(f"❌ Error: {e}")

@bot.command(name="deploy")
async def deploy_github(ctx, token: str, repo: str):
    uid = str(ctx.author.id)
    proj = user_data.get(uid, {}).get("current_project")
    if not proj: return await ctx.send("No active project.")
    proj_path = WORKSPACE_DIR / proj
    api_url = f"https://api.github.com/repos/{repo}/contents/"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    async with aiohttp.ClientSession() as session:
        for file_path in proj_path.rglob("*"):
            if file_path.is_file():
                relative = str(file_path.relative_to(proj_path))
                async with aiofiles.open(file_path, "rb") as f:
                    content_b64 = base64.b64encode(await f.read()).decode()
                payload = {"message": f"Deploy {relative}", "content": content_b64, "branch": "gh-pages"}
                try:
                    async with session.put(f"{api_url}{relative}", json=payload, headers=headers) as resp:
                        if resp.status not in (200, 201):
                            error_text = await resp.text()
                            return await ctx.send(f"GitHub error: {resp.status} - {error_text}")
                except Exception as e:
                    return await ctx.send(f"Deploy failed: {e}")
    await ctx.send(f"🚀 Deployed to https://{repo.split('/')[0]}.github.io/{repo.split('/')[1]}/")

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
            for item in WORKSPACE_DIR.iterdir():
                if item.is_file(): item.unlink()
            for uid in user_data: user_data[uid]["current_project"] = None
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
