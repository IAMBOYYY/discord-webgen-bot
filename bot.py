import os
import json
import asyncio
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler
from pathlib import Path

import discord
from discord.ext import commands
from openai import AsyncOpenAI, RateLimitError, APIStatusError

# -------------------------------------------------------------------
# Configuration from environment
# -------------------------------------------------------------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-2-9b-it:free")
PORT = int(os.getenv("PORT", 8000))
RETRY_LIMIT = 5               # max attempts
BASE_DELAY = 2                # seconds, doubles each retry

# -------------------------------------------------------------------
# Persistent user keys file
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
# HTTP server (health check + file serving)
# -------------------------------------------------------------------
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.directory = str(SITES_DIR)
            return SimpleHTTPRequestHandler.do_GET(self)

def start_http_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
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
    """Store your OpenRouter API key (free). Use in DM."""
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
# Core website generation with retry
# -------------------------------------------------------------------
@bot.command(name="make")
async def make_website(ctx, *, description: str):
    """Generate a website from text and host it publicly."""
    uid = str(ctx.author.id)
    api_key = get_user_key(uid)
    if not api_key:
        await ctx.send(
            "❌ You haven't set your OpenRouter API key yet.\n"
            "Get a free key at https://openrouter.ai/keys\n"
            "Then DM me: `!setkey your_key_here`"
        )
        return

    # Build OpenAI‑compatible client pointing to OpenRouter
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_API_BASE,
        default_headers={
            "HTTP-Referer": "https://discord.com",   # optional, OpenRouter likes it
            "X-Title": "Discord WebGen Bot"
        }
    )

    system_prompt = (
        "You are a web developer. Generate a complete single‑file HTML website "
        "with inline CSS and JavaScript. Reply ONLY with the raw HTML code "
        "(no markdown, no explanations). It must be a valid HTML5 document."
    )
    user_prompt = f"Create a website: {description}"

    # Retry loop for rate limits (429)
    for attempt in range(1, RETRY_LIMIT + 1):
        async with ctx.typing():
            try:
                response = await client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=4096,
                )
                html_code = response.choices[0].message.content.strip()
                # Strip code fences if present
                if html_code.startswith("```html"):
                    html_code = html_code[7:-3].strip()
                elif html_code.startswith("```"):
                    html_code = html_code[3:-3].strip()

                (SITES_DIR / "index.html").write_text(html_code, encoding="utf-8")
                render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")

                await ctx.send(
                    f"✅ Website generated!\n"
                    f"🌐 Public link: {render_url}\n"
                    f"🔄 Use `!make` again to update it."
                )
                return   # success, exit retry loop

            except RateLimitError as e:
                if attempt == RETRY_LIMIT:
                    await ctx.send("❌ Still hitting rate limits after several retries. Please try again later.")
                    return
                wait = BASE_DELAY * (2 ** (attempt - 1))  # 2, 4, 8, 16 sec...
                await ctx.send(f"⏳ Rate limited. Retrying in {wait} seconds… (attempt {attempt}/{RETRY_LIMIT})")
                await asyncio.sleep(wait)

            except APIStatusError as e:
                # Other API errors (like 400, 401, 500)
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
