$(document).ready(function() {
    $('#sendMsgButton').click(function(event) {
        event.preventDefault();
        var inputData = $('#chatbotInputField').val();// Prevent form submission and page reload
        
        // question to the div 
        chatelement = `<div class="user_question_div"> ${inputData}</div>`
        $('.chat-messages').append(chatelement);

        $.ajax({
            type: 'POST',
            url: '/button',
            data: { userinput: inputData },
            success: function(response) {
                // response to the div 
                chatelement = `<div class="llm_response_div"> ${response['llm_response']}</div>`
                $('.chat-messages').append(chatelement);
        
                console.log(response);
                // $('#message').text(response); // Display the response in the 'message' div
            },
            error: function() {
                console.log("fuck");
                // $('#message').text('Error: Request failed.');
            }
        });
    });
});