import pickle
import pandas as pd

with open('phishing.pk1','rb') as f:
    l_model = pickle.load(f)

with open('vectorizer.pk1','rb') as f:
    cv = pickle.load(f)

predict_bad=['yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php', 'fazan-pacir.rs/temp/libraries/ipad, www.tubemoviez.exe','svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt']
predict_good=['www.youtube.com/', 'youtube.com/watch?v=qI0TQJI3vdU, www.retailhellunderground.com/', 'restorevisioncenters.com/html/technology.html']

#FOR
predict_good_vector=cv.transform(predict_good)
predict_bad_vector=cv.transform(predict_bad)
print("Good URLs prediction:", l_model.predict(predict_good_vector))
print("Bad URLs prediction:", l_model.predict(predict_bad_vector))

