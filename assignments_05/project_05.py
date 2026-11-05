# -----------------------------------------------------------------------
# Mini-Project — Job Application Helper
# -----------------------------------------------------------------------

from dotenv import load_dotenv
from openai import OpenAI
import json

""" Task 1: Setup and System Prompt """

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

# System prompt for setting a role to AI assistant
system_prompt = """ You are a job application coach that helps users for career changes and job applications. 
        
        You can help with:
        - rewrite their resume bullet points, 
        - generate a draft cover letter

        Your behavioral guidelines:
        1. Stay focused on job application materials
        2. Always remind the user to review and edit its output before submitting anywhere
        3. Acknowledge that it may not know the user's specific industry norms, and that the user should use their own judgment     
        4. Ask follow-up questions
        5. Be encouraging and supportive, but not overly verbose
 """

if __name__ == "__main__":
    # Test that the system prompt and helper work as expected
    test_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Can you help me with career change and job applications? I'm switching from academia to industry."},
    ]

    reply = get_completion(test_messages)
    
    print("\n--- Task 1 Test prompt ---\n")
    print(reply)


# Add a comment explaining at least one deliberate choice you made in writing the system prompt and why.
# - I deliberately included a list of behavioral guidelines in the system prompt 
# - to ensure that the AI assistant remains focused on the user's job application needs.
# - I also emphasized the importance of user judgment and follow-up questions to make the interaction more personalized and effective.
# - I strictly speciafied the role and the tasks the AI assistant should perform to avoid any off-topic responses and to keep the conversation relevant to job applications.




""" Task 2: Bullet Point Rewriter """

def rewrite_bullets(bullets: list[str]) -> list[dict]:
    """Takes a list of resume bullet points, asks the model to improve them,
    and returns a list of dictionaries with 'original' and 'improved' keys.
    """
   
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    # Prompt for the model to rewrite the bullet points
    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list with NO additional text. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    
    # call get_completion()
    response = get_completion(messages)
    # added clening step as the model always responds with ```json and ``` at the start and end of the response, which is not valid JSON
    cleaned_response = (
            response.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        )
    # Parse the JSON response
    try:
        improved_bullets = json.loads(cleaned_response)
        return improved_bullets
    except json.JSONDecodeError:
        print("Error: The model's response was not valid JSON.")
        print(f"Raw output:\n{response}")
        return []

    
# Test starter bullets
if __name__ == "__main__":
    bullets = [
        "Helped customers with their problems",
        "Made reports for the management team",
        "Worked with a team to finish the project on time",
    ]

    print("\n--- Task 2: Resume Bullet Point Rewriter ---\n")
    improved_results = rewrite_bullets(bullets)

    for item in improved_results:
        print(f"Original: {item.get('original')}")
        print(f"Improved: {item.get('improved')}\n")
        
# What makes these bullets weak, and what kinds of changes did the model suggest?
# - The original bullets are weak because they are vague, lack specific achievements, and do not quantify results.
# - The model suggested changes that include specific actions, measurable outcomes, and stronger action verbs 
# - to make the bullet points more compelling and results-oriented. 


""" Task 3: Cover Letter Generator """

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]

    # Call get_completion() to generate the cover letter opening paragraph
    response = get_completion(messages)
    return response


# Test cover letter generation
if __name__ == "__main__":

    # sample job title and background for testing
    job_title = "Middle Data Engineer"
    background = "Ten years of experience as a research scientist and educator; recently completed \
        a Python course and built data pipelines using Pandas."
    
    # Call the function to generate the cover letter
    cover_letter = generate_cover_letter(job_title, background)
    print("\n--- Task 3: Cover Letter Generator ---\n")
    print(cover_letter)
    
# Why did you choose those particular examples? What does the few-shot pattern help control in the output?
# - I chose those examples because they are describe my background and the job title I am interested in to transition into.
# - The few-shot pattern helps control the output by providing clear examples of the desired style, tone, and structure for the cover letter opening paragraph. 


""" Task 4: Moderation Check """

def is_safe(text: str) -> bool:
    """Check if the text is safe for submission using OpenAI's moderation endpoint."""
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
   
    if flagged:
        print("The text contains content that may be inappropriate.")
        print("Categories flagged:", result.results[0].categories)
        return False
    else:
        return True

# Test moderation check
if __name__ == "__main__":
    print("\n--- Task 4: Moderation Check ---\n")
    
    test_text = "I love taking care of cats nd dogs."
    
    if is_safe(test_text):
        print("The text is safe for submission.\n")
    else:
        print("The text is NOT safe for submission.\n")

    test_text_2 = "I hate everyone on this planet and wish them all the worst."
    if is_safe(test_text_2):
        print("The text is safe for submission.")
    else:
        print("The text is NOT safe for submission.")




""" Task 5: The Chatbot Loop """

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: I can help you rewrite your bullet points!")
            print("Please paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # call rewrite_bullets() and print the results
            rewritten_bullets = rewrite_bullets(raw_bullets)
            print("\nJob Application Helper: Here are your original and improved bullet points:\n")
            print("-" * 50)
            for bullet in rewritten_bullets:
                print(f"Original: {bullet.get('original')}")
                print(f"Improved: {bullet.get('improved')}\n")
            print("-" * 50)
            print("\nJob Application Helper: Make sure to personalize these with specific examples and achievements!\n")
        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # call generate_cover_letter() and print the result
            if job_title and background:
                cover_letter = generate_cover_letter(job_title, background)
                print("\nJob Application Helper: Here's a draft opening paragraph for your cover letter:\n")
                print("-" * 50)
                print(cover_letter)
                print("-" * 50)
                print("\nJob Application Helper: Make sure to personalize this with the company's specific name and details!\n")
            else:
                print("Job Application Helper: Please provide both a job title and a brief background description.\n")  
        # 7. Otherwise, handle it as a regular chat turn
        else:
            # - Append the user's message to `messages`
            messages.append({"role": "user", "content": user_input})
            # - Call get_completion(messages)
            reply = get_completion(messages)
            # - Print the reply
            print(f"Job Application Helper: {reply}")
            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "assistant", "content": reply})
        # print(len(messages))    


if __name__ == "__main__":
    run_chatbot()


""" Task 6: Ethics Reflection """

# 1. Communication Styles
# - Large language models are predominantly trained on public web data originating from Western, corporate, and English-speaking contexts. 
# - Consequently, the bot naturally favors a direct, assertive, individual-centric communication style that is standard in North American tech hubs, 
# - often treating high self-promotion as the baseline. This can inadvertently disadvantage job-seekers from cultural backgrounds where humility, 
# - collective team achievement, or more modest phrasing are valued norms. 
# - Furthermore, the bot tends to default to corporate tech terminology and agile buzzwords, 
# - which may not align with conventions in non-profit, public sector, or traditional engineering environments.

# 2. Risks of Submitting Unreviewed Output:
# - If a candidate submits the bot's generated bullet points or cover letter without careful review, several critical issues could arise:
# - The model frequently invents plausible-sounding percentages, project scopes, or technical tooling that the applicant never actually worked with. 
# - If questioned during a technical interview, the candidate would be unable to defend these claims, damaging their credibility.
# - Generic placeholders: The model might leave brackets like "[Company Name]" or generic filler that immediately signals careless automation to recruiters.
# - Mismatched tone and robotic phrasing: Recruiters and hiring managers increasingly recognize standard LLM prose patterns 
# - (e.g., "fast-paced environment", "spearheaded cross-functional initiatives"), making the application appear disingenuous and disconnected from the candidate's authentic voice.

