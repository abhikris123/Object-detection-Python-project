import numpy as np
import os
import tensorflow as tf
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util

from tkinter import *
#from tkinter import messagebox
#from tkinter import filedialog


def send_email():
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    mail_content = 'Hello Sir,\n \nThere is a Robbery threat at the XYZ location. \nPlease take the necessary action. \n\nThanks'
    #The mail addresses and password
    sender_address = 'guitesting0506@gmail.com'
    sender_pass = 'Guitesting@0506'
    receiver_address = 'abhilashkrishna412@gmail.com'
    #Setup the MIME
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'High Alert: Robbery threat'   #The subject line
    #The body and the attachments for the mail
    message.attach(MIMEText(mail_content, 'plain'))
    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    


myWindow = Tk()
myWindow.geometry('500x400')
frame = Frame(myWindow, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH,expand=1)
frame.config(background='light blue')

PATH_TO_FROZEN_GRAPH = 'C:/Users/91986/Assessment3/Dataset/Runmodel/frozen_inference_graph.pb'

# path to the label map
PATH_TO_LABEL_MAP = 'C:/Users/91986/Assessment3/Dataset/Runmodel/label_map.pbtxt'

# number of classes 
NUM_CLASSES = 6



def webcam():
    cap = cv2.VideoCapture(0)
    #reads the frozen graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABEL_MAP)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    
# Detection
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
            # Read frame from camera
                ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
            # Extract image tensor
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Extract detection boxes
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Extract detection scores
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # Extract detection classes
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
            # Extract number of detections
                num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
                
            # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3,
                )
                
                send_email()
                
        # Display output
                cv2.imshow('Gesture Detection Page', cv2.resize(image_np, (1200, 800)))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break


myWindow.title("Gesture Controlled Trigger System")
openBtn1 = Button( frame, padx=5,pady=5,width=15, bg='white',fg='black', relief=GROOVE, text = "Webcam input", command = webcam,font=('helvetica 15 bold'))
#openBtn1.grid(column = 1 , row =25)
openBtn1.place(x=100,y=140)


myWindow.mainloop()

