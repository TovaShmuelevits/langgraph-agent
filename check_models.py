import os
import httpx
from groq import Groq

# מפתח ה-API שלך
os.environ["GROQ_API_KEY"] = ""

# עקיפת בדיקת SSL עבור סינון
custom_http_client = httpx.Client(verify=False)

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    http_client=custom_http_client
)

models = client.models.list()
print("המודלים הזמינים עבור המפתח שלך:")
for model in models.data:
    print(f"- {model.id}")
