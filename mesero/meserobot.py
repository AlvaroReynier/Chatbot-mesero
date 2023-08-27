import nltk
import tensorflow
import tflearn
import random 
import numpy as np
import pickle
import time
import json
import re
from frondend.chatwindow import *
from nltk.stem.lancaster import LancasterStemmer 

class Chatbot:
    stemmer =LancasterStemmer()
    hour =time.localtime().tm_hour
    menu =[{"pizza de peperoni": 13}, 
           {"pizza de queso": 12.40},
           {"pizza de jamon": 14.50},
           {"lasaña": 8},
           {"brownie": 5},
           {"banana split": 4},      
           {"coca-cola": 0.50},
           {"soda de fresa": 0.50}]
    total_del_pedido =0
    pedido =dict()
    ganancias =0
    with open("jsonchat.json",encoding='utf-8') as file:
        jsonchat_load =json.load(file)
    
    clientes ={"nombre":[],"apellido":[],"numero":[]}
    respuesta_anterior=""
    user_chats=[]
    bot_chats=[]
        
    def jsondata_preparation(user_text):
        words =[] 
        labels =[] 
        docs_x =[]
        docs_y =[]
        
        for intents in Chatbot.jsonchat_load['intents']:
            
            for patterns in intents['patterns']:
                wrds =nltk.word_tokenize(patterns) 
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intents["tag"])
                
                if intents['tag'] not in labels:
                    labels.append(intents['tag'])

        print(words)
        print("_"*90)        
        words =[Chatbot.stemmer.stem(w.lower()) for w in words if w != "?"]
        print(words)
        print("_"*90)   
        words =sorted(list(set(words)))    
        labels =sorted(labels)
        training =[]
        output =[]
        out_empty =[0 for _ in range (len(labels))]

        print(words)
        print("_"*90)
        print(labels)
        print("_"*95)
        print(docs_x)
        print("_"*95)
        print(docs_y)
        print("_"*95)
      
        for x, doc in enumerate(docs_x):
            bag =[]
            wrds =[Chatbot.stemmer.stem(w.lower()) for w in doc]
            
            for w in words:
                if w in wrds:                    
                    bag.append(1)
                    
                else:
                    bag.append(0)
                    
                output_row =out_empty[:]
                output_row[labels.index(docs_y[x])] = 1
                training.append(bag)
                output.append(output_row)
                
        training =np.array(training) 
        output =np.array(output) 
        Json =Chatbot.jsonchat_load
        
        with open ("data.pickle", "wb") as f:
            pickle.dump((words, labels,training,output,Json), f)
        
        return mesero_virtual.tranning_model(training,output,user_text=user_text)
        
    def tranning_model(training,output,user_text=None,model_net=None):
        tensorflow.compat.v1.reset_default_graph()        
        net = tflearn.input_data(shape=[None, len(training[0])]) 
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
        net = tflearn.regression(net)
        if model_net=='done':
            return net
        else:
            model = tflearn.DNN(net)
            model.fit(training, output, n_epoch=1024, batch_size=64, show_metric=True)
            model.save("model.tflearn")
            return mesero_virtual.answers(user_text,loop=1)
               

    
    def usertext_tansformation(user_text,words):
        bag =[0 for _ in range(len(words))]
        s_words =nltk.word_tokenize(user_text) 
        s_words =[Chatbot.stemmer.stem(word.lower()) for word in s_words]
        
        for se in s_words:
            
            for i, w in enumerate(words):
                
                if w == se:                    
                    bag[i] =1
                    
        return np.array(bag)
    
    def answers(user_text,loop=0):
        try:
            with open("data.pickle", "rb") as T:
                words, labels, training, output, Json =pickle.load(T)
                
            model =tflearn.DNN(mesero_virtual.tranning_model(training,output,model_net='done'))
            model.load('model.tflearn')
            
            if Json!=Chatbot.jsonchat_load:
                raise("el archivo json fue modificado, preparando nuevo archivo")
                
        except Exception as e: 
            print("error",e)
            if loop==0:
                return mesero_virtual.jsondata_preparation(user_text)
            
            else:
                print("error de carga")
                
        else:
            results =model.predict([mesero_virtual.usertext_tansformation(user_text.lower(), words)])
            results_index =np.argmax(results)        
            tag =labels[results_index]
            Chatbot.user_chats.append(user_text.lower())
            index = next(index for (index, intent) in enumerate(Chatbot.jsonchat_load["intents"]) if intent["tag"] == tag)
            
            print(index) 
            if re.search(r'\b(no|eso seria todo|nada)\b', user_text.lower()) and Chatbot.respuesta_anterior=="desea algo mas":
                pe =""
                print(Chatbot.pedido)
                print(Chatbot.total_del_pedido)
                for x, y in Chatbot.pedido.items():
                    pe =pe+str(y)+" "+str(x)+"\n"
                Chatbot.respuesta_anterior="desea confirmar la orden"
                responses=str(pe)+"Total del pedido es de:"+str(Chatbot.total_del_pedido)+"\n"+str(Chatbot.respuesta_anterior)

            elif Chatbot.respuesta_anterior=="desea confirmar la orden":
                if re.search(r'\b(si)\b', user_text.lower()):
                    responses="ingrese nombre, apellido y numero telefonico para validar la compra(ejemplo:Enrique Martinez 9852-4508)"
                    Chatbot.respuesta_anterior="validacion"
                elif re.search(r'\b(no|cancelar)\b', user_text.lower()):
                    responses="el pedido a sido cancelado"
                    Chatbot.total_del_pedido=0
                    Chatbot.pedido=dict()
                    Chatbot.respuesta_anterior=""
                else:
                    responses="porfavor, confirme la orden"
                
            elif Chatbot.respuesta_anterior=="validacion":
                datos_cliente=re.compile('^(\w+) (\w+) (\d{8}|\d{4}-\d{4})$')
                match=datos_cliente.search(user_text)
                if match:
                    Chatbot.clientes["nombre"].append(match.group(1))
                    Chatbot.clientes["apellido"].append(match.group(2))
                    Chatbot.clientes["numero"].append(match.group(3))
                    responses ="Total del pedido es de:"+str(Chatbot.total_del_pedido)+"\npedido validado, gracias por su compra"
                    Chatbot.ganancias =Chatbot.total_del_pedido+Chatbot.ganancias
                    Chatbot.total_del_pedido =0
                    Chatbot.pedido =dict()
                    Chatbot.respuesta_anterior =""
                elif re.search(r'\b(no|cancelar)\b', user_text.lower()):
                    responses="el pedido a sido cancelado"
                    Chatbot.total_del_pedido=0
                    Chatbot.pedido=dict()
                    Chatbot.respuesta_anterior=""
                else:
                    responses =("el formato usado no es valido porfavor ingrese sus datos nuevamente(ejemplo:Enrique Martinez 9852-4508)")
                   
                           
            elif Chatbot.jsonchat_load["intents"][index]['tag']=="Español-saludos":
                if Chatbot.hour < 12:
                    responses ="Buenos días"+str(random.choice(Chatbot.jsonchat_load["intents"][index]['responses']))
                        
                elif Chatbot.hour < 18:
                    responses ="Buenas tardes"+str(random.choice(Chatbot.jsonchat_load["intents"][index]['responses']))
                        
                else:
                    responses ="Buenas noches"+str(random.choice(Chatbot.jsonchat_load["intents"][index]['responses']))
                        
            elif Chatbot.jsonchat_load["intents"][index]['tag']=="Español-menu":
                numeros = re.findall(r'\d+|otra|otro|un|una|dos|de', user_text.lower())
                claves = re.findall('(pizza|lasaña|brow|banana|fresa|coca|^de)', user_text.lower())
                
                for x in numeros:
                    if x=="de" and Chatbot.respuesta_anterior =="que pizza" and re.search(r'^\d+|\b(^un|^una)\b', user_text.lower()) is None:
                        print("hola")
                        numeros[numeros.index(x)] =1
                    elif x=="de":
                        numeros.remove(x)               
                print(numeros)
                for x in numeros:
                    if x=="otra" or x=="otro" or x=="una" or x=="un":
                        numeros[numeros.index(x)] =1
                    if x=="dos":
                        numeros[numeros.index(x)] =2
                        
                for x in claves:    
                    if x=="de" and Chatbot.respuesta_anterior =="que pizza" :
                        claves[claves.index(x)]="pizza"
                    elif x=="de":
                        claves.remove(x)
                        
                for x in claves:                        
                    if x=="otra" or x=="otro":
                        claves[claves.index(x)]=claves[claves.index(x)-1]

                        
                pizz=0
                        
                pizzas =re.findall('(peperoni|jamon|queso)', user_text.lower())
                f =0
                for x in claves:                
                    if x=="pizza":
                        try:
                            claves[claves.index(x)] =x+" de "+str(pizzas[f])
                            f+=1
                        except:
                            pizz=1
                        else:
                            f+=1
                            
                    elif x=="banana":
                        claves[claves.index(x)] =x+" split"
                        
                    elif x=="fresa":
                        claves[claves.index(x)] ="soda de "+x 
                        
                    elif x=="coca":
                        claves[claves.index(x)] ="coca-cola"
                        
                    elif x=="brow":
                        claves[claves.index(x)] ="brownie"
                        
                for num, palabra in zip(numeros, claves):
                    Chatbot.pedido[palabra] =int(num)
                print(Chatbot.pedido)
                print(Chatbot.total_del_pedido)
                Chatbot.total_del_pedido =0
                    
                for producto, cantidad in Chatbot.pedido.items():
                     for item in Chatbot.menu:
                        if producto.lower() in item:
                            Chatbot.total_del_pedido +=cantidad * item[producto.lower()]
                if pizz==0:
                    responses ="desea algo mas"
                    Chatbot.respuesta_anterior =responses
                else:
                    responses ="de que le gustaria su pizza"
                    Chatbot.respuesta_anterior ="que pizza"                    
                         
                
            elif Chatbot.jsonchat_load["intents"][index]['tag']=="Español-tipos-comida":
        
                if re.search('(pizza|lasaña|postre|bebida|tomar)', user_text.lower()):                    
                    tipos_de =re.findall('(pizza|lasaña|postre|bebida|tomar)', user_text.lower())

                    if tipos_de[0]=="pizza":
                        pizza_tipos =list(filter(lambda x: list(x.keys())[0].find("pizza") != -1, Chatbot.menu))
                        pizza_tipos =[e for x in pizza_tipos for e in x]
                        t=""
                        for x in pizza_tipos:
                            t=t+str(x)+", "
                        
                        responses="tenemos: "+str(t)
                    
                    elif tipos_de[0]=="lasaña":
                        responses ="tenemos lasaña de carne"
                    
                    elif tipos_de[0]=="postre":                
                        responses ="tenemos brownie y banana split"
                    
                    elif tipos_de[0]=="bebida" or tipos_de[0]=="tomar":                
                        responses ="tenemos soda de fresa y coca-cola" 
                else:  
                    responses="no tenemos actualmente"
                    
            else:

                responses =random.choice(Chatbot.jsonchat_load["intents"][index]['responses'])

            Chatbot.bot_chats.append(responses)
            print(Chatbot.user_chats)
            print(Chatbot.bot_chats)
            print(Chatbot.clientes)
            print(Chatbot.ganancias)              
            return responses
            
mesero_virtual=Chatbot  

chat(mesero_virtual)



