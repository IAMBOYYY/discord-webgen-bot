import os
import json
import asyncio
import threading
import shutil
import uuid
import time
import base64
import io
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

# Chat providers
PROVIDERS = {
    "openrouter": {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "default_model": "google/gemma-2-9b-it"},
    "groq": {"name": "Groq", "base_url": "https://api.groq.com/openai/v1", "default_model": "llama-3.1-8b-instant"},
    "nvidia": {"name": "NVIDIA NIM", "base_url": "https://integrate.api.nvidia.com/v1", "default_model": "meta/llama-3.1-8b-instruct"},
    "mistral": {"name": "Mistral AI", "base_url": "https://api.mistral.ai/v1", "default_model": "mistral-small-latest"},
    "cerebras": {"name": "Cerebras", "base_url": "https://api.cerebras.ai/v1", "default_model": "llama3.1-8b"},
    "vertex": {"name": "Vertex AI", "base_url": os.getenv("VERTEX_BASE_URL", ""), "default_model": "gemini-1.5-flash"},
}

# Image providers
IMAGE_PROVIDERS = {
    "openrouter": {"name": "OpenRouter images", "base_url": "https://openrouter.ai/api/v1", "default_model": "stabilityai/sd3-turbo", "api_type": "openai_images"},
    "pollinations": {"name": "Pollinations AI", "base_url": "https://image.pollinations.ai/prompt", "default_model": "flux", "api_type": "pollinations"},
    "replicate": {"name": "Replicate", "base_url": "https://api.replicate.com/v1", "default_model": "stability-ai/sdxl", "api_type": "replicate"},
}

# -------------------------------------------------------------------
# User data
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

@bot.event
async def on_ready():
    print(f"✅ Logged in as {bot.user}")
    start_http_server()
    await startup_cleanup()

# -------------------------------------------------------------------
# Global error handler (shows user what went wrong)
# -------------------------------------------------------------------
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return
    await ctx.send(f"❌ Error: {str(error)}")
    print(f"Error in {ctx.command}: {error}")

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
        try:
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                models = data.get("data", data.get("models", []))
                return [m["id"] for m in models if "id" in m]
        except:
            return []

async def fetch_image_models(provider: str, api_key: str) -> list:
    if provider == "openrouter":
        prov = PROVIDERS["openrouter"]
        url = f"{prov['base_url']}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
                    return [m["id"] for m in data.get("data", []) if "image" in m.get("capabilities", [])]
            except:
                return []
    return [IMAGE_PROVIDERS[provider]["default_model"]]

async def show_all_models(ctx, models: list):
    if not models:
        await ctx.send("No models found.")
        return
    total = len(models)
    lines = [f"`{i+1}` - `{m}`" for i, m in enumerate(models)]
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
**Available Commands**
`!setup` – Configure chat + image generation
`!models` – Switch chat model
`!setkey <key>` – Update API key
`!provider` – Show current provider/model
`!newproject` – Create new website project
`!listfiles` – List files in project
`!viewfile <filename>` – View file content
`!setproject <name>` – Switch active project
`!listprojects` – List all projects
`!make <description>` – Build website/code
`!edit <file> <instruction>` – Edit a file with AI
`!attach [image] <description>` – Build from image (retries on credit errors)
`!draw <prompt>` – Generate an image
`!ask <question>` – Ask anything (server‑aware)
`!search <query>` – Web search with AI filter
`!imgsetup` – Configure image generation only
`!imgmodels` – Switch image model
`!devcleanup` – Admin wipe (password required)
Multi‑command: use ` && ` between commands.
""")

# -------------------------------------------------------------------
# !setup (full, works in DMs)
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

        models_sorted = sorted(models)
        await show_all_models(ctx, models_sorted)
        await ctx.send("Reply with number or exact model ID. 5 min timeout.")

        def model_check(m):
            return m.author == ctx.author and m.channel == ctx.channel
        msg = await bot.wait_for("message", timeout=300.0, check=model_check)
        choice_text = msg.content.strip()
        if choice_text.isdigit():
            idx = int(choice_text)
            if 1 <= idx <= len(models_sorted):
                chosen_model = models_sorted[idx-1]
            else:
                return await ctx.send("Invalid number. Setup cancelled.")
        else:
            if choice_text in models_sorted:
                chosen_model = choice_text
            else:
                return await ctx.send("Model ID not found. Setup cancelled.")

        user_data[uid] = {"provider": provider, "api_key": api_key, "model": chosen_model, "current_project": None}
        save_user_data()
        await ctx.send(f"✅ Chat setup complete!\nProvider: **{prov_cfg['name']}**\nModel: `{chosen_model}`")

        # Image setup prompt
        await ctx.send("Do you want to set up **image generation**? (yes/no)")
        def yes_no_check(m):
            return m.author == ctx.author and m.channel == ctx.channel and m.content.lower() in ("yes", "no")
        try:
            img_choice = await bot.wait_for("message", timeout=180.0, check=yes_no_check)
        except asyncio.TimeoutError:
            return await ctx.send("Skipping image setup. Use `!imgsetup` later.")
        if img_choice.content.lower() == "no":
            return await ctx.send("Image generation not configured. Use `!imgsetup` later.")

        # Image provider selection
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

    except asyncio.TimeoutError:
        await ctx.send("⌛ Setup timed out.")

# -------------------------------------------------------------------
# !imgsetup (works in DMs)
# -------------------------------------------------------------------
@bot.command(name="imgsetup")
async def img_setup(ctx):
    uid = str(ctx.author.id)
    if uid not in user_data or "api_key" not in user_data[uid]:
        return await ctx.send("Run `!setup` first.")
    await ctx.send("Now configure image generation.")
    img_keys = list(IMAGE_PROVIDERS.keys())
    lines = ["**Choose image provider:**"] + [f"`{i}` - {IMAGE_PROVIDERS[k]['name']}" for i, k in enumerate(img_keys, 1)]
    await ctx.send("\n".join(lines))
    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel and m.content.isdigit()
    try:
        msg = await bot.wait_for("message", timeout=180.0, check=check)
        choice = int(msg.content)
        if choice < 1 or choice > len(img_keys):
            return await ctx.send("Invalid choice.")
        img_provider = img_keys[choice-1]
        img_cfg = IMAGE_PROVIDERS[img_provider]
        await ctx.send(f"Selected **{img_cfg['name']}**. Enter API key (or `skip` to use existing key).")
        key_msg = await bot.wait_for("message", timeout=180.0, check=lambda m: m.author == ctx.author and m.channel == ctx.channel)
        api_key = key_msg.content.strip()
        if api_key.lower() == "skip" and img_provider == "openrouter" and "api_key" in user_data[uid]:
            api_key = user_data[uid]["api_key"]
        elif api_key.lower() == "skip":
            return await ctx.send("No key available.")
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
        def model_check(m):
            return m.author == ctx.author and m.channel == ctx.channel
        msg = await bot.wait_for("message", timeout=300.0, check=model_check)
        choice_text = msg.content.strip()
        chosen = None
        if choice_text.isdigit():
            idx = int(choice_text)
            if 1 <= idx <= len(models):
                chosen = models[idx-1]
        else:
            if choice_text in models:
                chosen = choice_text
        if not chosen:
            return await ctx.send("Invalid selection.")
        user_data[uid]["image_provider"] = img_provider
        user_data[uid]["image_api_key"] = api_key
        user_data[uid]["image_model"] = chosen
        save_user_data()
        await ctx.send(f"✅ Image gen ready: `{chosen}`")
    except asyncio.TimeoutError:
        await ctx.send("⌛ Timed out.")

# -------------------------------------------------------------------
# !imgmodels (works in DMs)
# -------------------------------------------------------------------
@bot.command(name="imgmodels")
async def list_img_models(ctx):
    uid = str(ctx.author.id)
    if uid not in user_data or "image_provider" not in user_data[uid]:
        return await ctx.send("No image gen configured.")
    provider = user_data[uid]["image_provider"]
    api_key = user_data[uid]["image_api_key"]
    await ctx.send("🔍 Fetching image models…")
    models = await fetch_image_models(provider, api_key)
    if not models:
        return await ctx.send("Could not fetch models.")
    await show_all_models(ctx, models)
    await ctx.send("Reply with number or model ID to switch (or `cancel`).")
    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel
    try:
        msg = await bot.wait_for("message", timeout=300.0, check=check)
        if msg.content.lower() == "cancel":
            return
        chosen = None
        if msg.content.isdigit():
            idx = int(msg.content)
            if 1 <= idx <= len(models):
                chosen = models[idx-1]
        elif msg.content in models:
            chosen = msg.content
        if chosen:
            user_data[uid]["image_model"] = chosen
            save_user_data()
            await ctx.send(f"✅ Switched to `{chosen}`.")
        else:
            await ctx.send("Invalid selection.")
    except asyncio.TimeoutError:
        await ctx.send("Timed out.")

# -------------------------------------------------------------------
# !draw (no file stored, prompt enhancement, clear status)
# -------------------------------------------------------------------
@bot.command(name="draw")
async def draw_image(ctx, *, prompt: str):
    uid = str(ctx.author.id)
    if uid not in user_data or "image_provider" not in user_data[uid]:
        return await ctx.send("❌ Image generation not configured. Use `!imgsetup`.")
    provider = user_data[uid]["image_provider"]
    api_key = user_data[uid]["image_api_key"]
    model = user_data[uid].get("image_model", IMAGE_PROVIDERS[provider]["default_model"])
    cfg = IMAGE_PROVIDERS[provider]

    # Enhance prompt (fallback to original)
    enhanced_prompt = prompt
    client = get_user_client(uid)
    if client:
        try:
            chat_model = get_user_model(uid)
            resp = await client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": "You are a prompt engineer. Turn the user's short description into a detailed, high-quality image generation prompt. Output only the prompt."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200,
            )
            enhanced_prompt = resp.choices[0].message.content.strip()
            await ctx.send(f"✨ Enhanced prompt: {enhanced_prompt}")
        except Exception as e:
            await ctx.send(f"⚠️ Prompt enhancement failed ({e}), using original prompt.")
    else:
        await ctx.send("⚠️ Chat client not set up; using original prompt.")

    await ctx.send("🎨 Generating image...")
    async with ctx.typing():
        try:
            if provider == "openrouter":
                image_client = AsyncOpenAI(api_key=api_key, base_url=cfg["base_url"])
                resp = await image_client.images.generate(model=model, prompt=enhanced_prompt, n=1, size="1024x1024")
                await ctx.send(f"🖼️ {resp.data[0].url}")
            elif provider == "pollinations":
                url = f"{cfg['base_url']}/{enhanced_prompt}?model={model}&key={api_key}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as img_resp:
                        if img_resp.status == 200:
                            img_data = await img_resp.read()
                            file = discord.File(io.BytesIO(img_data), filename="image.png")
                            await ctx.send(file=file)
                        else:
                            await ctx.send(f"Pollinations error: HTTP {img_resp.status}")
            elif provider == "replicate":
                headers = {"Authorization": f"Token {api_key}"}
                async with aiohttp.ClientSession() as session:
                    payload = {"version": model, "input": {"prompt": enhanced_prompt}}
                    async with session.post(f"{cfg['base_url']}/predictions", json=payload, headers=headers) as resp:
                        if resp.status != 201:
                            return await ctx.send("Replicate error.")
                        data = await resp.json()
                    pred_id = data["id"]
                    await asyncio.sleep(5)
                    async with session.get(f"{cfg['base_url']}/predictions/{pred_id}", headers=headers) as resp:
                        data = await resp.json()
                    if data["status"] == "succeeded":
                        await ctx.send(f"🖼️ {data['output'][0]}")
                    else:
                        await ctx.send("Still processing, check later.")
            else:
                await ctx.send("Unsupported provider.")
        except Exception as e:
            if "404" in str(e) and "No model found" in str(e):
                await ctx.send(f"❌ Model `{model}` not found. Use `!imgmodels` to see available models.")
            else:
                await ctx.send(f"❌ Image generation failed: {e}")

# -------------------------------------------------------------------
# !attach (with retry on credit limit)
# -------------------------------------------------------------------
@bot.command(name="attach")
async def attach_image(ctx, *, description: str = ""):
    if not ctx.message.attachments:
        return await ctx.send("Attach an image with `!attach`.")
    uid = str(ctx.author.id)
    if uid not in user_data:
        return await ctx.send("Set up provider first.")
    client = get_user_client(uid)
    if not client:
        return await ctx.send("Provider error.")
    model = get_user_model(uid)
    attachment = ctx.message.attachments[0]
    if attachment.size > 500_000:
        await ctx.send("⚠️ Image is large; consider compressing to under 500KB.")

    # Try to switch to a vision model if needed
    if "vision" not in model.lower() and "gemini" not in model.lower() and "gpt-4" not in model.lower():
        models = await fetch_models(user_data[uid]["provider"], user_data[uid]["api_key"])
        vision_candidates = [m for m in models if "vision" in m.lower() or "gemini" in m.lower() or "gpt-4" in m.lower()]
        if vision_candidates:
            model = vision_candidates[0]
            await ctx.send(f"ℹ️ Using vision model `{model}`.")
        else:
            return await ctx.send("No vision model available.")

    image_bytes = await attachment.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": f"Build a website based on this image. {description}"},
        {"type": "image_url", "image_url": {"url": f"data:{attachment.content_type};base64,{image_b64}"}}
    ]}]

    # Retry up to 3 times with 30s wait on credit errors
    for attempt in range(3):
        async with ctx.typing():
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=256,   # extremely low to avoid credit blowout
                )
                ai_plan = resp.choices[0].message.content
                return await ctx.invoke(bot.get_command("make"), description=ai_plan)
            except Exception as e:
                error_str = str(e)
                if "402" in error_str and "credits" in error_str:
                    if attempt == 2:
                        return await ctx.send("❌ Still out of credits after 3 attempts. Please try a smaller image or add credits.")
                    await ctx.send(f"⏳ Credit limit hit, waiting 30 seconds (attempt {attempt+1}/3)...")
                    await asyncio.sleep(30)
                else:
                    return await ctx.send(f"❌ Error: {e}")

# -------------------------------------------------------------------
# Other commands (!models, !setkey, !provider, !ask, projects, !make, !edit, !search, multi-command, devcleanup)
# (All remaining commands unchanged, included for completeness)
# -------------------------------------------------------------------
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
    await show_all_models(ctx, sorted(models))
    await ctx.send("Reply with number or model ID to switch (or `cancel`).")
    def check(m):
        return m.author == ctx.author and m.channel == ctx.channel
    try:
        msg = await bot.wait_for("message", timeout=300.0, check=check)
        if msg.content.lower() == "cancel":
            return
        chosen = None
        if msg.content.isdigit():
            idx = int(msg.content)
            if 1 <= idx <= len(models):
                chosen = sorted(models)[idx-1]
        elif msg.content in models:
            chosen = msg.content
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
    if uid not in user_data:
        return await ctx.send("Run `!setup` first.")
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

# -------------------------------------------------------------------
# !ask (server-aware)
# -------------------------------------------------------------------
SERVER_KEYWORDS = ["server", "guild", "member", "owner", "moderator", "admin", "channel", "role", "emojis", "boost", "level", "created", "this place", "here", "how many people", "who owns", "who is online", "online members"]

def is_server_question(q: str) -> bool:
    return any(word in q.lower() for word in SERVER_KEYWORDS)

def get_server_info(guild: discord.Guild) -> str:
    total = guild.member_count
    online_members = [m.name for m in guild.members if m.status != discord.Status.offline]
    online_str = ", ".join(online_members[:10])
    if len(online_members) > 10:
        online_str += f" and {len(online_members)-10} more"
    roles = ", ".join([r.name for r in guild.roles[:10]])
    owner = guild.owner.name if guild.owner else "Unknown"
    created = guild.created_at.strftime("%B %d, %Y")
    return (
        f"Server: {guild.name}\nOwner: {owner}\nTotal members: {total}\nOnline ({len(online_members)}): {online_str}\n"
        f"Text channels: {len(guild.text_channels)}, Voice: {len(guild.voice_channels)}\nRoles: {roles}\nCreated: {created}"
    )

@bot.command(name="ask")
async def ask_question(ctx, *, question: str):
    uid = str(ctx.author.id)
    client = get_user_client(uid)
    if not client:
        return await ctx.send("❌ Set up provider first.")
    model = get_user_model(uid)
    messages = []
    if ctx.guild and is_server_question(question):
        messages.append({"role": "system", "content": f"Server data:\n{get_server_info(ctx.guild)}\nAnswer using this data."})
    else:
        messages.append({"role": "system", "content": "You are a helpful assistant."})
    messages.append({"role": "user", "content": question})
    async with ctx.typing():
        try:
            resp = await client.chat.completions.create(model=model, messages=messages, temperature=0.7, max_tokens=1024)
            await ctx.send(resp.choices[0].message.content[:2000])
        except Exception as e:
            await ctx.send(f"❌ Error: {e}")

# -------------------------------------------------------------------
# Project commands
# -------------------------------------------------------------------
@bot.command(name="newproject")
async def new_project(ctx):
    uid = str(ctx.author.id)
    proj = f"proj_{uuid.uuid4().hex[:8]}"
    (WORKSPACE_DIR / proj).mkdir(parents=True, exist_ok=True)
    workspace_meta[proj] = time.time()
    save_workspace_meta()
    if uid not in user_data:
        user_data[uid] = {}
    user_data[uid]["current_project"] = proj
    save_user_data()
    asyncio.create_task(schedule_project_deletion(proj))
    render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
    await ctx.send(f"📁 Project `{proj}`\n🌐 {render_url}/{proj}/")

@bot.command(name="listfiles")
async def list_files_cmd(ctx):
    uid = str(ctx.author.id)
    proj = user_data.get(uid, {}).get("current_project")
    if not proj:
        return await ctx.send("No active project.")
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
    if not proj:
        return await ctx.send("No active project.")
    file_path = WORKSPACE_DIR / proj / filename
    if not file_path.exists():
        return await ctx.send("File not found.")
    async with aiofiles.open(file_path, "r") as f:
        content = await f.read()
    if len(content) > 2000:
        await ctx.send(file=discord.File(file_path, filename=filename))
    else:
        await ctx.send(f"**{filename}**\n```{filename.split('.')[-1]}\n{content}\n```")

@bot.command(name="setproject")
async def set_project(ctx, name: str):
    uid = str(ctx.author.id)
    if not (WORKSPACE_DIR / name).exists():
        return await ctx.send("Project not found.")
    if uid not in user_data:
        user_data[uid] = {}
    user_data[uid]["current_project"] = name
    save_user_data()
    await ctx.send(f"✅ Active project: `{name}`")

@bot.command(name="listprojects")
async def list_projects(ctx):
    render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
    if not workspace_meta:
        return await ctx.send("No projects.")
    lines = ["**Projects:**"]
    for folder in workspace_meta:
        lines.append(f"• `{folder}` → {render_url}/{folder}/")
    await ctx.send("\n".join(lines))

# -------------------------------------------------------------------
# !make
# -------------------------------------------------------------------
@bot.command(name="make")
async def make_website(ctx, *, description: str):
    uid = str(ctx.author.id)
    if uid not in user_data or not user_data[uid].get("current_project"):
        return await ctx.send("❌ No active project. Use `!newproject`.")
    client = get_user_client(uid)
    if not client:
        return await ctx.send("Set up provider first.")
    model = get_user_model(uid)
    proj_name = user_data[uid]["current_project"]
    proj_path = WORKSPACE_DIR / proj_name

    tools = [
        {"type": "function", "function": {"name": "create_file", "description": "Create a file", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}, "content": {"type": "string"}}, "required": ["filename", "content"]}}},
        {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}}},
        {"type": "function", "function": {"name": "list_files", "description": "List files", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "delete_file", "description": "Delete a file", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}}},
    ]

    system_prompt = (
        "You are an expert developer. Build as requested.\n"
        "- For websites: separate HTML/CSS/JS, use Unsplash for images (`https://source.unsplash.com/random/800x600/?topic`).\n"
        "- For games: make them fully playable.\n"
        "- For other languages: create appropriate files.\n"
        "- Multi-page: create separate HTML files.\n"
        "- After all file operations, output `DONE:` followed by a summary.\n"
        "- JSON must be valid."
    )

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Build: {description}"}]
    progress_msg = await ctx.send("🤖 Building...")
    async with ctx.typing():
        try:
            for _ in range(10):
                resp = await client.chat.completions.create(model=model, messages=messages, tools=tools, tool_choice="auto", temperature=0.7)
                msg = resp.choices[0].message
                messages.append(msg)
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        try:
                            args = json.loads(tc.function.arguments)
                        except:
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": "Invalid JSON, retry."})
                            continue
                        func = tc.function.name
                        if func == "create_file":
                            filename = args["filename"]
                            content = args["content"]
                            (proj_path / filename).parent.mkdir(parents=True, exist_ok=True)
                            async with aiofiles.open(proj_path / filename, "w") as f:
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
            final_files = [str(p.relative_to(proj_path)) for p in proj_path.rglob("*") if p.is_file()]
            render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
            await progress_msg.edit(content=f"🌐 {render_url}/{proj_name}/\n**Files:**\n" + "\n".join(f"• `{f}`" for f in final_files) + "\n✅ Build complete.")
            workspace_meta[proj_name] = time.time()
            save_workspace_meta()
            asyncio.create_task(schedule_project_deletion(proj_name))
        except Exception as e:
            await progress_msg.edit(content=f"❌ Error: {e}")

# -------------------------------------------------------------------
# !edit
# -------------------------------------------------------------------
@bot.command(name="edit")
async def edit_file(ctx, filename: str, *, instruction: str):
    uid = str(ctx.author.id)
    proj = user_data.get(uid, {}).get("current_project")
    if not proj:
        return await ctx.send("No active project.")
    file_path = WORKSPACE_DIR / proj / filename
    if not file_path.exists():
        return await ctx.send("File not found.")
    async with aiofiles.open(file_path, "r") as f:
        original = await f.read()
    client = get_user_client(uid)
    if not client:
        return await ctx.send("Set up provider first.")
    model = get_user_model(uid)
    async with ctx.typing():
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Edit the file as instructed. Return ONLY the new content."},
                    {"role": "user", "content": f"File: {filename}\nContent:\n{original}\n\nInstruction: {instruction}"}
                ],
                temperature=0.2,
            )
            new_content = resp.choices[0].message.content
            async with aiofiles.open(file_path, "w") as f:
                await f.write(new_content)
            await ctx.send(f"✅ `{filename}` updated. Use `!viewfile {filename}` to see.")
        except Exception as e:
            await ctx.send(f"❌ Error: {e}")

# -------------------------------------------------------------------
# !search
# -------------------------------------------------------------------
@bot.command(name="search")
async def search_web(ctx, *, query: str):
    uid = str(ctx.author.id)
    client = get_user_client(uid)
    if not client:
        return await ctx.send("Set up provider first.")
    url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return await ctx.send("Search failed.")
                data = await resp.json()
                abstract = data.get("AbstractText", "")
                related = data.get("RelatedTopics", [])
                snippets = [abstract] if abstract else []
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
        {"role": "system", "content": "Use the search results to answer concisely, filtering irrelevant info."},
        {"role": "user", "content": f"Question: {query}\n\nResults:\n{search_text}"}
    ]
    async with ctx.typing():
        try:
            resp = await client.chat.completions.create(model=model, messages=messages, temperature=0.5, max_tokens=500)
            await ctx.send(resp.choices[0].message.content[:2000])
        except Exception as e:
            await ctx.send(f"❌ Error: {e}")

# -------------------------------------------------------------------
# Multi-command
# -------------------------------------------------------------------
@bot.event
async def on_message(message):
    if message.author.bot:
        return
    if message.content.startswith("!") and " && " in message.content:
        parts = [p.strip() for p in message.content.split(" && ")]
        for part in parts:
            if part:
                original = message.content
                message.content = part
                await bot.process_commands(message)
                message.content = original
        return
    await bot.process_commands(message)

# -------------------------------------------------------------------
# Admin cleanup (now also deletes any temp files in workspace)
# -------------------------------------------------------------------
@bot.command(name="devcleanup")
async def dev_cleanup(ctx):
    try:
        await ctx.author.send("🔐 Enter admin password:")
    except discord.Forbidden:
        return await ctx.send("I cannot DM you.")
    def check(m):
        return m.author == ctx.author and isinstance(m.channel, discord.DMChannel)
    try:
        msg = await bot.wait_for("message", timeout=30.0, check=check)
        if msg.content == ADMIN_PASSWORD:
            # Delete all projects
            for folder in list(workspace_meta.keys()):
                await delete_project(folder)
            # Delete any stray files in workspace root
            for item in WORKSPACE_DIR.iterdir():
                if item.is_file():
                    item.unlink()
            for uid in user_data:
                user_data[uid]["current_project"] = None
            save_user_data()
            await ctx.author.send("🗑️ All projects and temporary files wiped.")
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
