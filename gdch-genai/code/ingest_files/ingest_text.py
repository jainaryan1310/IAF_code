import json
import os

from llm_factory import gemini_generate, ollama_generate, ollama_create
from loguru import logger

system_prompt = """Extract the following **entities**, **relations** and **properties** from the provided text:

### **Entities:**
1. **Individuals:** Names of people.
2. **Places:** Names of locations (cities, countries, specific addresses, etc.).
3. **Organisation:** Names of businesses, groups, or organizations.
4. **Personal Details:** Phone numbers, identification numbers, addresses, registration numbers, etc.
5. **Booking Details:** Flight numbers, train ticket numbers, hotel names, travel time, etc.
6. **Particular Objects:** Specific references to personal or unique objects (e.g., a particular car, a bag, a watch).
7. **Events:** References to activities such as meetings, deals, parties, vacations, etc.

### **Relations:**
1. **Location:** Indicates where an individual, company, or object is located or where an event took place. 
2. **Ownership:** Links an individual to objects, personal details, or other items they possess.
3. **Usage:** Describes how an individual uses objects (devices, vehicles, etc.) or personal details (phone numbers, tickets, etc.).
4. **Call:** Indicates that one individual contacted another via a phone call.
5. **Message:** Indicates that one individual communicated with another via a text message or email.
6. **Participate:** Shows that an individual took part in an event.
7. **Member:** Establishes that an individual is a member of an organization or company.

### **Properties:**
Descriptions or adjectives associated with any entity, including roles (e.g., "manager"), physical characteristics (e.g., "black car"), or other qualifying details (e.g., a specific date or time).

### **Example Output:**

For the text:

"Emily Johnson, a software developer at TechSolutions, attended a conference in San Francisco on September 20th, 2024. She drove her white BMW 3 Series to the airport and parked it at Terminal 2. At the conference, Emily met with her manager, Daniel Harris, and discussed their upcoming project, 'Apollo'. After the meeting, she booked a room at the Grand Bay Hotel through Booking.com for two nights. On the same evening, Emily texted her friend, Kevin Lee, about the successful meeting. Daniel, who owns a Samsung Galaxy S21, called Emily later to go over some details. Emily also reserved a seat on Flight AA123 with American Airlines, scheduled for September 22nd, 2024."

Response:
{
  "entities": [
    { "type": "individual", "name": "Emily Johnson" },
    { "type": "individual", "name": "Daniel Harris" },
    { "type": "individual", "name": "Kevin Lee" },
    { "type": "place", "name": "San Francisco" },
    { "type": "place", "name": "Terminal 2" },
    { "type": "org", "name": "TechSolutions" },
    { "type": "org", "name": "American Airlines" },
    { "type": "org", "name": "Booking.com" },
    { "type": "object", "name": "White BMW 3 Series" },
    { "type": "object", "name": "Samsung Galaxy S21" },
    { "type": "event", "name": "Conference" },
    { "type": "event", "name": "Meeting" },
    { "type": "event", "name": "Project 'Apollo'" },
    { "type": "booking_detail", "name": "Room at Grand Bay Hotel" },
    { "type": "booking_detail", "name": "Flight AA123" }
  ],
  "relations": [
    {
      "entity1": "Emily Johnson",
      "relation": "location",
      "entity2": "San Francisco"
    },
    {
      "entity1": "Emily Johnson",
      "relation": "ownership",
      "entity2": "White BMW 3 Series"
    },
    {
      "entity1": "Daniel Harris",
      "relation": "member",
      "entity2": "TechSolutions"
    },
    {
      "entity1": "Emily Johnson",
      "relation": "participate",
      "entity2": "Conference"
    },
    {
      "entity1": "Emily Johnson",
      "relation": "usage",
      "entity2": "White BMW 3 Series"
    },
    {
      "entity1": "Daniel Harris",
      "relation": "ownership",
      "entity2": "Samsung Galaxy S21"
    },
    {
      "entity1": "Emily Johnson",
      "relation": "call",
      "entity2": "Daniel Harris"
    },
    {
      "entity1": "Emily Johnson",
      "relation": "message",
      "entity2": "Kevin Lee"
    },
    {
      "entity1": "Emily Johnson",
      "relation": "participate",
      "entity2": "Meeting"
    },
    {
      "entity1": "Emily Johnson",
      "relation": "booking",
      "entity2": "Room at Grand Bay Hotel"
    },
    {
      "entity1": "Emily Johnson",
      "relation": "booking",
      "entity2": "Flight AA123"
    }
  ],
  "properties": [
    {
      "entity": "Emily Johnson",
      "description": "software developer"
    },
    {
      "entity": "Daniel Harris",
      "description": "manager"
    },
    {
      "entity": "White BMW 3 Series",
      "description": "car"
    },
    {
      "entity": "Samsung Galaxy S21",
      "description": "phone"
    },
    {
      "entity": "Conference",
      "description": "September 20th, 2024"
    },
    {
      "entity": "Room at Grand Bay Hotel",
      "description": "two nights"
    },
    {
      "entity": "Flight AA123",
      "description": "September 22nd, 2024"
    }
  ]
}


"""


def ingest_text(source: str, file_name: str, output_folder: str):
    if os.environ["LLM"] == "mistral":
        return ingest_text_ollama(source, file_name, output_folder)
    else:
        return ingest_text_gemini(source, file_name, output_folder)


def ingest_text_gemini(source: str, file_name: str, output_folder: str):
    with open(source + file_name, "r") as f:
        text = f.read()

    response_json = gemini_generate(system_prompt, [text])

    if response_json["code"] != 200:
        logger.error(f"Error during response generation for file {source + file_name}")
        return response_json

    try:
        response = response_json["response"]
        logger.info(response)

        response = response.replace("```json", "")
        response = response.replace("```", "")
        response = response.replace("\n", "")

        info = json.loads(response)
        info["code"] = 200

        logger.info(info)

        with open(output_folder+file_name, "w", encoding="utf-8") as f:
            json.dump({"info": info}, f, ensure_ascii=False, indent=4)

        return info

    except Exception as e:
        logger.error(f"There was an error in parsing the response. Response : {response}. Error : {e}")
        return response_json

    
def ingest_text_ollama(source: str, file_name: str, output_folder: str):
    ollama_create()
    with open(source + file_name, "r") as f:
        text = f.read()

    response = ollama_generate(system_prompt + text)
    print(response)

    try:

        response = response.replace("\n", "")
        info = json.loads(response)
        info["code"] = 200

        logger.info(info)

        with open(output_folder+file_name, "w", encoding="utf-8") as f:
            json.dump({"info": info}, f, ensure_ascii=False, indent=4)

        return info

    except Exception as e:

        logger.error(f"There was an error in parsing the response. Response : {response}. Error : {e}")
        return response