import base64

with open("test.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")
print(b64)
