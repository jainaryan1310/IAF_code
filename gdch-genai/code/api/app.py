from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt

# import requests

from test import find_best_matching
from execute import crop_image, classify_color, get_average_color, increase_contrast
import constants

from google.cloud import storage, firestore  # type: ignore
from io import BytesIO
from datetime import date

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from agent import agent

app = Flask(__name__)

LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = "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"
LANGCHAIN_PROJECT = "gdch-genai-flask"

# creating a firestore database client
db = firestore.Client(project="niti-aayog-410004")
collection_name = constants.COLLECTION_NAME

# Changing thte flask CORS config
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    expose_headers="*",
    supports_credentials=True,
)
app.config["SESSION_COOKIE_SAMESITE"] = "None"
app.config["SESSION_COOKIE_SECURE"] = True


# Check if the provided token is active and extract user_id from active token
def check_active(request):
    auth = request.headers.get("Authorization").split()
    if auth and len(auth) == 2:
        token = auth[1]
        try:
            payload = jwt.decode(token, "flos seges humilis", algorithms=["HS256"])
            return str(payload["id"])
        except:
            return 0


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)


@app.route("/")
def default():
    return "The API server is running!"


@app.route("/authenticate/")
# authenticate if the received token is valid and active
def authenticate():
    user_id = check_active(request)

    if user_id == 0:
        response = jsonify({"auth": False, "response": "Unauthorised Access"})
        return response, 401

    response = jsonify({"auth": True, "response": "Authorised Access"})
    return response


@app.route("/start_session/")
# start a new user session
def start_session():
    user_id = check_active(request)
    if user_id == 0:
        response = jsonify({"response": "Unauthorised Access"})
        return response, 401

    response = jsonify({"user_id": user_id, "response": "A Chat session has started"})
    return response


@app.route("/new_chat/")
# start a new chat document in the chat database
def new_chat():
    # check token authenticity
    user_id = check_active(request)
    if user_id == 0:
        response = jsonify({"response": "Unauthorised Access"})
        return response, 401

    query = db.collection(collection_name).where("user_id", "==", user_id)
    docs = query.get()

    ucid = len(docs) + 1
    doc_id = f"{user_id}_{ucid}"

    chat_log = {
        "user_id": user_id,
        "ucid": ucid,
        "title": "new chat",
        "date": date.today().strftime("%d/%m/%y"),
        "message_history": [],
        "deleted": False,
    }

    doc_ref = db.collection(collection_name).document(doc_id)
    doc_ref.set(chat_log)

    response = jsonify(
        {
            "user_id": user_id,
            "ucid": ucid,
            "response": "A new chat has been initialised",
        }
    )
    return response


@app.route("/edit_title/")
# edit the display title of a chat in the history
def edit_title():
    # check token authenticity
    user_id = check_active(request)
    if user_id == 0:
        response = jsonify({"response": "Unauthorised Access"})
        return response, 401

    ucid = int(request.args.get("ucid"))
    title = request.args.get("title")

    doc_id = f"{user_id}_{ucid}"
    chat_doc = db.collection(collection_name).document(doc_id)

    new_title = {"title": title}
    chat_doc.set(new_title, merge=True)

    response = jsonify(
        {
            "user_id": user_id,
            "ucid": ucid,
            "title": title,
            "response": "The chat title has been updated",
        }
    )

    return response


@app.route("/get_chats/")
# get the list of all old chats by this user
def get_chats():
    # check token authenticity
    user_id = check_active(request)
    if user_id == 0:
        response = jsonify({"response": "Unauthorised Access"})
        return response, 401

    # querye the db for all old chats for given user
    query = db.collection(collection_name).where("user_id", "==", user_id)
    docs = query.get()

    doc_list = []
    for doc in docs:
        d = doc.to_dict()
        if d["message_history"] != [] and d["deleted"] == False:
            # return chat_title instead of first message | also return date in response
            doc_list.append({"ucid": d["ucid"], "date": d["date"], "title": d["title"]})

    response = jsonify({"user_id": user_id, "doc_list": doc_list})
    return response


@app.route("/restore_chat/")
# restore and continue an old chat
def restore_chat():
    # check token authenticity
    user_id = check_active(request)
    if user_id == 0:
        response = jsonify({"response": "Unauthorised Access"})
        return response, 401

    ucid = int(request.args.get("ucid"))
    doc_id = f"{user_id}_{ucid}"

    doc = db.collection(collection_name).document(doc_id)
    chat_log = doc.get().to_dict()

    response = jsonify(chat_log)
    return response


@app.route("/get_response/")
# get response for a new message
def get_response():
    # check token authenticity
    user_id = check_active(request)
    if user_id == 0:
        response = jsonify({"response": "Unauthorised Access"})
        return response, 401

    user_input = request.args.get("user_input")
    ucid = int(request.args.get("ucid"))
    language = request.args.get("language")
    language_api = request.args.get("api")
    service = request.args.get("service")
    doc_id = f"{user_id}_{ucid}"

    chat_doc = db.collection(collection_name).document(doc_id)
    chat_history = chat_doc.get().to_dict()['message_history']

    if chat_history == []:
        chat_doc.set({"title": user_input}, merge=True)

    response = agent.chat(user_input)
    
    chat_history.append({"author": "user", "content": user_input, "source": "None", "translated_content": "None"})
    chat_history.append({"author": "bot", "content": response, "source": "None", "translated_content": "None"})

    message_history = {"message_history": chat_history}
    chat_doc.set(message_history, merge=True)

    message_history = jsonify(message_history)
    return message_history
    

@app.route("/delete_chat/")
# soft delete a chat from the db
def delete_chat():
    # check token authenticity
    user_id = check_active(request)
    if user_id == 0:
        response = jsonify({"response": "Unauthorised Access"})
        return response, 401

    ucid = int(request.args.get("ucid"))
    doc_id = f"{user_id}_{ucid}"

    deleted = {"deleted": True}

    chat_doc = db.collection(collection_name).document(doc_id)
    chat_doc.set(deleted, merge=True)

    response = jsonify({"user_id": user_id, "ucid": ucid, "response": "The chat has been deleted"})
    return response


@app.route("/like_response/")
# user likes the reponse as feedback
def like_response():
    # check token authenticity
    user_id = check_active(request)
    if user_id == 0:
        response = jsonify({"response": "Unauthorised Access"})
        return response, 401

    ucid = int(request.args.get("ucid"))
    msg_id = int(request.args.get("msg_id"))
    doc_id = f"{user_id}_{ucid}"

    chat_doc = db.collection(collection_name).document(doc_id)
    chat_history = chat_doc.get().to_dict()['message_history']

    chat_history[msg_id]["like"] = True

    # update chat history in chat db
    message_history = {"message_history": chat_history}
    chat_doc.set(message_history, merge=True)

    response = jsonify({"user_id": user_id, "ucid": ucid, "msg_id": msg_id, "response": "The like was noted"})
    return response


@app.route("/dislike_response/")
# user dislikes the reponse as feedback
def dislike_response():
    # check token authenticity
    user_id = check_active(request)
    if user_id == 0:
        response = jsonify({"response": "Unauthorised Access"})
        return response, 401

    ucid = int(request.args.get("ucid"))
    msg_id = int(request.args.get("msg_id"))
    doc_id = f"{user_id}_{ucid}"

    chat_doc = db.collection(collection_name).document(doc_id)
    chat_history = chat_doc.get().to_dict()['message_history']

    chat_history[msg_id]["like"] = False

    # update chat history in chat db
    message_history = {"message_history": chat_history}
    chat_doc.set(message_history, merge=True)

    response = jsonify({"user_id": user_id, "ucid": ucid, "msg_id": msg_id, "response": "The dislike was noted"})
    return response


@app.route("/comment/")
# user comments on the reponse as feedback
def comment():
    # check token authenticity
    user_id = check_active(request)
    if user_id == 0:
        response = jsonify({"response": "Unauthorised Access"})
        return response, 401

    ucid = int(request.args.get("ucid"))
    msg_id = int(request.args.get("msg_id"))
    comment = request.args.get("comment")
    doc_id = f"{user_id}_{ucid}"

    chat_doc = db.collection(collection_name).document(doc_id)
    chat_history = chat_doc.get().to_dict()['message_history']

    chat_history[msg_id]["comment"] = comment

    # update chat history in chat db
    message_history = {"message_history": chat_history}
    chat_doc.set(message_history, merge=True)

    response = jsonify({"user_id": user_id, "ucid": ucid, "msg_id": msg_id, "response": "The comment was noted"})
    return response


@app.route("/suggested_response/")
# user suggests a response for the last query
def suggestion():
    # check token authenticity
    user_id = check_active(request)
    if user_id == 0:
        response = jsonify({"response": "Unauthorised Access"})
        return response, 401

    ucid = int(request.args.get("ucid"))
    msg_id = int(request.args.get("msg_id"))
    suggested_response = request.args.get("suggested_response")
    doc_id = f"{user_id}_{ucid}"

    chat_doc = db.collection(collection_name).document(doc_id)
    chat_history = chat_doc.get().to_dict()['message_history']

    chat_history[msg_id]["suggested_response"] = suggested_response

    # update chat history in chat db
    message_history = {"message_history": chat_history}
    chat_doc.set(message_history, merge=True)

    response = jsonify(
        {"user_id": user_id, "ucid": ucid, "msg_id": msg_id, "response": "The suggested_response was noted"})
    return response



@app.route("/face_recognition", methods=["POST"])
def face_recognition():
    user_id = check_active(request)

    if user_id == 0:
        response = jsonify({"auth": False, "response": "Unauthorised Access"})
        return response, 401

    if request.method == "POST":
        try:
            # image_name = request.form.get('image_path')
            image_name = request.form.get("image_path")
            image_cont_name = (
                "valiance-face-recognition"  # Example image container name
            )

            folder_cont_name = (
                "valiance-face-recognition"  # Example folder container name
            )
            folder_name = "output"  # Example folder name where the images are stored

            result = find_best_matching(
                image_name, image_cont_name, folder_cont_name, folder_name
            )

            return jsonify(
                {
                    "success": True,
                    "message": "Image processed successfully.",
                    "result": result,
                }
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            return jsonify({"error": str(e)})


@app.route("/car_colour", methods=["POST"])
def car_colour():
    if request.method == "POST":
        try:
            image_path = request.form.get("image_path")
            image = calculate_brightness("valiance-car-colour", image_path)
            if image is not None:
                contrast_image = increase_contrast(image)
                cropped_image = crop_image(contrast_image)
                mean_color = get_average_color(cropped_image)
                color_name = classify_color(mean_color)
                result = (
                    f"The dominant color of the car in {image_path} is: {color_name}"
                )
            else:
                result = f"Failed to load image: {image_path}"

            return jsonify(
                {
                    "success": True,
                    "message": "Image processed successfully.",
                    "result": result,
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)})


def calculate_brightness(bucket_name, image_blob_name):

    # Initialize the GCS client
    client = storage.Client()

    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(image_blob_name)

    image_data = BytesIO(blob.download_as_bytes())

    if image_data is None:
        return None, None

    image_array = np.frombuffer(image_data.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        print("Error: Unable to decode the image")
        return None, None

    return np.array(image, dtype=np.uint8)
