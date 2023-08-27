from tkinter import *
import tkinter as TK

def chat(mesero_virtual):
    ventana = Tk()
    ventana.title("Mesero-virtual")
    ventana.geometry("400x500")
    ventana.resizable(width=FALSE, height=FALSE)
    ventana.config(bg='#212121')
    ChatLog = Text(ventana, bd=0, bg="#332f2c", height="8", width="50", font="tahoma")
    ChatLog.place(x=6, y=6, height=386, width=370)
    scrollbar = Scrollbar(ventana, command=ChatLog.yview)
    ChatLog['yscrollcommand']=scrollbar.set
    scrollbar.place(x=376, y=6, height=386)
    ChatLog.config(state=DISABLED)
    EntryBox = Text(ventana, bd=0, bg="#6b6b6b",foreground="white", width="29", height="5", font="Arial")
    EntryBox.place(x=6, y=401, height=90, width=265)
        
    def send():
        user_text=EntryBox.get("1.0", "end-1c").strip()
        EntryBox.delete("0.0",END)           
        answer = mesero_virtual.answers(user_text)
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, user_text+'\n\n',"chat")
        ChatLog.tag_config("chat",foreground="white", font=("tahoma",16))
        ChatLog.insert(END, answer+'\n\n',"user")
        ChatLog.tag_config("user",foreground="#0082c1", font=("tahoma",16))
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
            
    SendButton = Button(ventana, font=("verdana", 12, 'bold'), text="Send", width=9,
                            height=5, bd=0, bg="#001a57", activebackground="gold", 
                            fg='white', command=send)
    SendButton.place(x=282, y=401, height=90)
    ventana.bind('<Return>', lambda event:send())
    ventana.mainloop()
        
    
if __name__ == "__main__":
    chat()
