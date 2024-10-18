css = '''
<style> 
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #E9ECEF
}
.chat-message.bot {
    background-color: #E9ECEF
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 68px;
  max-height: 68px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #212529;
}
'''

logo = '''
<div style= margin-bottom: 15px;">
    <img src="https://i.ibb.co/Sckp999/Imagen-de-Whats-App-2024-10-16-a-las-13-28-07-4def5707.jpg" alt="Logo" style="max-width: 20%; height: 20%;">
</div>
'''


bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/GQYcqB2/3d-illustration-cartoon-business-woman-character-avatar-profile-1183071-533.jpg">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.ibb.co/0BPSWZy/japanese-suit-upscaleds.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''