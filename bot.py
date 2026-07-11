import os
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler
from pathlib import Path

import discord
from discord.ext import commands
from openai import AsyncOpenAI

# -------------------------------------------------------------------
# Configuration from environment variables
# -------------------------------------------------------------------
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
SAKANA_API_BASE = os.getenv("SAKANA_API_BASE", "https://api.sakana.ai/v1")
SAKANA_MODEL = os.getenv("SAKANA_MODEL", "fugu")
# No global SAKANA_API_KEY required – each user sets their own

# File for user API keys
USER_KEYS_FILE = Path("user_keys.json")
user_keys = {}

# Generated sites folder
SITES_DIR = Path("generated_sites")
SITES_DIR.mkdir(exist_ok=True)

# Port for the built‑in HTTP server (Render expects a PORT env var, but we use 8000)
PORT = int(os.getenv("PORT", 8000))

# -------------------------------------------------------------------
# Load / save user keys
# -------------------------------------------------------------------
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
# Combined HTTP server: serves generated sites AND a health endpoint
# -------------------------------------------------------------------
class HealthHandler(BaseHTTPRequestHandler):
    """Handler that returns 200 OK for /health (used by Render)."""
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            # For all other paths, fall back to serving files from SITES_DIR
            SimpleHTTPRequestHandler.do_GET(self)

def start_http_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    server.directory = str(SITES_DIR)  # set the base directory for file serving
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
        await ctx.send("✅ API key saved. For safety, use DMs next time.", delete_after=10)
    else:
        await ctx.send("✅ API key saved securely.")

@bot.command(name="make")
async def make_website(ctx, *, description: str):
    uid = str(ctx.author.id)
    api_key = get_user_key(uid)
    if not api_key:
        await ctx.send("❌ You haven't set your Sakana API key. Use `!setkey YOUR_KEY` in a DM.")
        return

    async with ctx.typing():
        try:
            client = AsyncOpenAI(api_key=api_key, base_url=SAKANA_API_BASE)
            system_prompt = (
                "You are a web developer. Generate a complete single‑file HTML website "
                "with inline CSS and JS. Reply ONLY with raw HTML (no markdown)."
            )
            response = await client.chat.completions.create(
                model=SAKANA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Create a website: {description}"},
                ],
                temperature=0.7,
                max_tokens=4096,
            )
            html_code = response.choices[0].message.content.strip()
            if html_code.startswith("```html"):
                html_code = html_code[7:-3].strip()
            elif html_code.startswith("```"):
                html_code = html_code[3:-3].strip()

            (SITES_DIR / "index.html").write_text(html_code, encoding="utf-8")

            # Get the public URL for the site from Render's environment
            render_url = os.getenv("RENDER_EXTERNAL_URL", f"http://localhost:{PORT}")
            await ctx.send(
                f"✅ Website generated!\n"
                f"🌐 Public link: {render_url}\n"
                f"🔄 Use `!make` again to update it."
            )
        except Exception as e:
            await ctx.send(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("❌ DISCORD_TOKEN is not set.")
        exit(1)
    bot.run(DISCORD_TOKEN)
