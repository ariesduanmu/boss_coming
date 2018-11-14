# Boss_coming

Since I have a working project recently, face recognize is needed.

So I just wanna **practice** more on face recongnize, by detecting boss's face and pretend being hard worker haha


Three ways to do the face recognition

1. face encoding + face compare from `face_recogition` 

Good side: easy to use

Bad side: each face(even same people) is one data, so that when I have huge image data, and there is one new face come in, it will need lots time to do the **compare**

2. ML

Good side: accuracy

Bad side: tough, and tbh I don't know which is more accurant, compare fucntion in face_recognition or ML.

3. center of encoded data(kmeans maybe)

Here is my "image"

I got encoded data and labeled, then measure the center of these data(throw bad data)

and there comes a new image, I just compare it with those centers

