a. **Accident Detection**: Utilize the YOLOv7 model, incorporate accident photos, and use LabelImg for annotation to train the model. Once an accident is detected, the system will capture images of the involved vehicles and return the location information.

b. **Cross-Border Vehicle Tracking**: Use location information to search for nearby surveillance cameras and retrieve footage from a few seconds before the incident. Perform vehicle reidentification to find the vehicle with the highest match and proceed to license plate recognition.

c. **License Plate Recognition for Accident Vehicles**: Use the YOLOv7 model, incorporate license plate photos, and use LabelImg for annotation to train the model. Perform license plate recognition on the vehicle identified through cross-border tracking.

d. **Data Transmission to Police and Relevant Parties**: Vehicle owners and relevant parties need to download the "ShieldMyRide" app, register, and enter their information. This way, when an accident occurs, the accident location can be sent to the relevant personnel.

  `pip install -r requirements.txt`<br>
  `python caryolo_2.py`
