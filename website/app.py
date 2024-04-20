from flask import Flask, render_template, request
from rag_orch import doRAG

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/button', methods=['POST'])
def button():
    user_input = request.form['userinput']
    
    # rag_response = {
    #     "llm_response": "test things",
    #     "pagelabel": 5
    # }
    
    rag_response = doRAG(user_input)
    
    
    # Here, you can add the functionality you want to execute when the button is pressed.
    return rag_response

if __name__ == '__main__':
    app.run(debug=True)