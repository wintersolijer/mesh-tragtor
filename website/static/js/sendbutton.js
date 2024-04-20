$(document).ready(function() {
    $('#sendMsgButton').click(function(event) {
        event.preventDefault();
        var inputData = $('#chatbotInputField').val();// Prevent form submission and page reload
        
        // question to the div 
        chatelement = `<div class="user_question_div"> ${inputData}</div>`
        $('.chat-messages').append(chatelement);
        $('#chatbotInputField').val("");

        $.ajax({
            type: 'POST',
            url: '/button',
            data: { userinput: inputData },
            success: function(response) {
                // response to the div 
                link_text = `https://www.trumpf.com/filestorage/TRUMPF_Master/Products/Power_Electronics/Energy_storage/TRUMPF_Manual_TruConvert_DC_1030.pdf#page=${response['pagelabel'][0]}`

                chat_element = `<div class="llm_response_div"> ${response['llm_response']}<br><a href="${link_text}">Click here for more information</a></div>`
                

                $('.chat-messages').append(chat_element);
        
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