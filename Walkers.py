import cv2


body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')


cap = cv2.VideoCapture('walking.avi')


while True:
    
    ret, frame = cap.read()
    
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)
    
    
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    
    cv2.imshow('Pedestrian Detection', frame)

    if cv2.waitKey(1) == 32: 
        break

cap.release()
cv2.destroyAllWindows()
