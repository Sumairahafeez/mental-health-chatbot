import google.generativeai as genai


def test_gemini():
    genai.configure(api_key="AIzaSyBq1zON_vbXdkKWmJcr_oc59NnxbSg76CM")

    model = genai.GenerativeModel('gemini-2.0-flash')

    prompt = "Tell me a fun fact about the moon."

    print("Sending request to Gemini...")

    try:
        response = model.generate_content(contents=[prompt])

        print("Response received")

        if response.text is None:
            print("Error from Gemini API:", response.error)
        else:
            print("Generated text:", response.text)
    except Exception as e:
        print("Exception while calling Gemini:", str(e))

if __name__ == "__main__":
    test_gemini()
