import os
import json
import asyncio
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import discord
from discord.ext import commands
from openai import AsyncOpenAI, RateLimitError, APIStatusError
import aiohttp

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "auto")
PORT = int(os.getenv("PORT", 8000))
RETRY_LIMIT = 5
BASE_DELAY = 2

# HTML quality
MIN_HTML_LENGTH = 100
REQUIRED_TAGS = ["<html", "<head", "<body"]

# -------------------------------------------------------------------
# Persistent user keys
# -------------------------------------------------------------------
USER_KEYS_FILE = Path("user_keys.json")
user_keys = {}

def load_user_keys():
    global user_keys
    if USER_KEYS_FILE.exists():
        with open(USER_KEYS_FILE, "r") as f:
            user_keys = json.load(f)
    else:
        user_keys = {}

def save_user_keys():
    with open(USER_KEYS_FILE, "w") as f:
        json.dump(user_keys, f, indent=2)

load_user_keys()

# -------------------------------------------------------------------
# Generated sites folder
# -------------------------------------------------------------------
SITES_DIR = Path("generated_sites")
SITES_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Auto model selection
# -------------------------------------------------------------------
MODEL_PRIORITY = [
    "google/gemma-2-9b-it",
    "meta-llama/llama-3-8b-instruct",
    "mistralai/mistral-7b-instruct",
    "meta-llama/llama-3-8b",
    "mistralai/mistral-7b",
    "google/gemini-2.0-flash-001",
]
FALLBACK_MODEL = "google/gemma-2-9b-it"

selected_model = None
model_lock = asyncio.Lock()

async def fetch_best_free_model(api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://discord.com",
        "X-Title": "Discord WebGen Bot"
    }
    url = f"{OPENROUTER_API_BASE}/models"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, timeout=10) as resp:
                if resp.status != 200:
                    raise Exception(f"Models API returned {resp.status}")
                data = await resp.json()
        except Exception as e:
            print(f"Failed to fetch models: {e}")
            return FALLBACK_MODEL

    free_models = []
    for model in data.get("data", []):
        pricing = model.get("pricing", {})
        prompt_price = pricing.get("prompt", "0")
        completion_price = pricing.get("completion", "0")
        try:
            if float(prompt_price) == 0 and float(completion_price) == 0:
                free_models.append(model["id"])
        except (ValueError, KeyError):
            continue

    for model_id in MODEL_PRIORITY:
        if model_id in free_models:
            print(f"Auto-selected model: {model_id}")
            return model_id

    if free_models:
        fallback = free_models[0]
        print(f"No priority model found, using first free: {fallback}")
        return fallback

    return FALLBACK_MODEL

# -------------------------------------------------------------------
# HTTP server
# -------------------------------------------------------------------
class RequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITES_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            super().do_GET()

def start_http_server():
    server = HTTPServer(("0.0.0.0", PORT), RequestHandler)
    print(f"🌐 Web server running on port {PORT}")
    threading.Thread(target=server.serve_forever, daemon=True).start()

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

def get_user_key(user_id: str) -> str | None:
    return user_keys.get(str(user_id))

@bot.command(name="setkey")
async def set_key(ctx, key: str):
    uid = str(ctx.author.id)
    user_keys[uid] = key
    save_user_keys()
    if ctx.guild:
        try:
            await ctx.message.delete()
        except:
            pass
        await ctx.send("✅ OpenRouter API key saved. For safety, use DMs next time.", delete_after=10)
    else:
        await ctx.send("✅ OpenRouter API key saved securely.")

# -------------------------------------------------------------------
# HTML validation
# -------------------------------------------------------------------
def is_valid_html(html: str) -> bool:
    if len(html) < MIN_HTML_LENGTH:
        return False
    html_lower = html.lower()
    for tag in REQUIRED_TAGS:
        if tag not in html_lower:
            return False
    return True

# -------------------------------------------------------------------
# Website generation
# -------------------------------------------------------------------
@bot.command(name="make")
async def make_website(ctx, *, description: str):
    uid = str(ctx.author.id)
    api_key = get_user_key(uid)
    if not api_key:
        await ctx.send(
            "❌ You haven't set your OpenRouter API key yet.\n"
            "Get a free key at https://openrouter.ai/keys\n"
            "Then DM me: `!setkey your_key_here`"
        )
        return

    global selected_model
    if OPENROUTER_MODEL == "auto" and selected_model is None:
        async with model_lock:
            if selected_model is None:
                await ctx.send("🔍 Finding the best free model for you…", delete_after=5)
                selected_model = await fetch_best_free_model(api_key)
                print(f"Auto model set to: {selected_model}")

    model_to_use = OPENROUTER_MODEL if OPENROUTER_MODEL != "auto" else selected_model

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_API_BASE,
        default_headers={
            "HTTP-Referer": "https://discord.com",
            "X-Title": "Discord WebGen Bot"
        }
    )

    # Base system prompt
    system_prompt = (
        "You are a professional web developer. Generate a COMPLETE, single‑file HTML website "
        "with visible text content, CSS styling, and proper structure. "
        "Rules:\n"
        "- Include <!DOCTYPE html>, <html>, <head> with <title> and meta charset.\n"
        "- Use inline CSS to make the page attractive (dark text on light bg or similar).\n"
        "- The <body> MUST contain at least one <h1> and one <p> with meaningful content.\n"
        "- Do NOT output an empty page or only a script. The page must render content without JavaScript.\n"
        "- Output ONLY the raw HTML. No markdown, no explanations."
    )

    user_prompt = f"Create a website: {description}"

    for gen_attempt in range(3):
        for attempt in range(1, RETRY_LIMIT + 1):
            async with ctx.typing():
                try:
                    response = await client.chat.completions.create(
                        model=model_to_use,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.7,
                        max_tokens=4096,
                    )

                    # Extract content safely
                    content = response.choices[0].message.content
                    if content is None:
                        raise ValueError("Model returned empty content (None).")

                    html_code = content.strip()

                    # Clean code fences
                    if html_code.startswith("```html"):
                        html_code = html_code[7:-3].strip()
                    elif html_code.startswith("```"):
                        html_code = html_code[3:-3].strip()

                    # Validate
                    if is_valid_html(html_code):
                        (SITES_DIR / "index.html").write_text(html_code, encoding="utf-8")
                        render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
                        await ctx.send(
                            f"✅ Website generated!\n"
                            f"🌐 Public link: {render_url}\n"
                            f"🔄 Use `!make` again to update it."
                        )
                        return
                    else:
                        # Invalid HTML – retry with stricter prompt
                        if gen_attempt == 2:
                            await ctx.send("❌ After several attempts, the page was still blank. Please try a more specific description.")
                            return
                        await ctx.send("⚠️ Page appears blank, retrying with clearer instructions…")
                        user_prompt = f"Create a full, visible website with at least one heading and one paragraph about: {description}"
                        break  # exit inner loop, go to next gen_attempt

                except ValueError as ve:
                    # Empty content – treat as quality failure
                    if gen_attempt == 2:
                        await ctx.send("❌ The model repeatedly returned empty content. Try a different description or model.")
                        return
                    await ctx.send("⚠️ Model returned empty response, retrying…")
                    user_prompt = f"Please output ONLY valid HTML code for a website about: {description}"
                    break

                except RateLimitError:
                    if attempt == RETRY_LIMIT:
                        await ctx.send("❌ Still hitting rate limits. Please try again later.")
                        return
                    wait = BASE_DELAY * (2 ** (attempt - 1))
                    await ctx.send(f"⏳ Rate limited. Retrying in {wait} seconds… (attempt {attempt}/{RETRY_LIMIT})")
                    await asyncio.sleep(wait)

                except APIStatusError as e:
                    await ctx.send(f"❌ API error: {e.status_code} – {e.message}")
                    return

                except Exception as e:
                    await ctx.send(f"❌ Unexpected error: {str(e)}")
                    return

# -------------------------------------------------------------------
# Start bot
# -------------------------------------------------------------------
if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("❌ DISCORD_TOKEN not set.")
        exit(1)
    bot.run(DISCORD_TOKEN)
