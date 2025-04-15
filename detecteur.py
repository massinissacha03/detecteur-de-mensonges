import cv2
import numpy as np
import mediapipe as mp
import time
from scipy.spatial import distance as dist

# Initialisation
cap = cv2.VideoCapture("test.MP4")  # Remplace par le nom exact de ta vidéo
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
signal = []
bpm_values = []
last_bpm_time = time.time()
blink_count = 0
last_blink_time = time.time()
lie_scores = []  # Pour lisser le score
tells = {}
calibration_frames = 0
calibrated = False
MAX_FRAMES = 120
RECENT_FRAMES = 12
EYE_BLINK_HEIGHT = 0.15
SIGNIFICANT_BPM_CHANGE = 8
LIP_COMPRESSION_RATIO = 0.35
TELL_MAX_TTL = 30
FACEMESH_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]

# Optionnel : décommente pour enregistrer
# out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))

def decrement_tells(tells):
    for key, tell in tells.copy().items():
        if 'ttl' in tell:
            tell['ttl'] -= 1
            if tell['ttl'] <= 0:
                del tells[key]
    return tells

def new_tell(result):
    return {'text': result, 'ttl': TELL_MAX_TTL}

def get_aspect_ratio(top, bottom, right, left):
    height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
    width = dist.euclidean([right.x, right.y], [left.x, left.y])
    return height / width if width > 0 else 0

def is_blinking(face):
    eyeR = [face[p] for p in [159, 145, 133, 33]]
    eyeR_ar = get_aspect_ratio(*eyeR)
    eyeL = [face[p] for p in [386, 374, 362, 263]]
    eyeL_ar = get_aspect_ratio(*eyeL)
    eyeA_ar = (eyeR_ar + eyeL_ar) / 2
    return eyeA_ar < EYE_BLINK_HEIGHT

def get_blink_tell(blinks):
    if sum(blinks[:RECENT_FRAMES]) < 3:
        return None
    recent_closed = 1.0 * sum(blinks[-RECENT_FRAMES:]) / RECENT_FRAMES
    avg_closed = 1.0 * sum(blinks) / MAX_FRAMES
    if recent_closed > (20 * avg_closed):
        return "Increased blinking"
    elif avg_closed > (20 * recent_closed):
        return "Decreased blinking"
    return None

def check_hand_on_face(hands_landmarks, face):
    if hands_landmarks:
        face_landmarks = [face[p] for p in FACEMESH_FACE_OVAL]
        face_points = [[[p.x, p.y] for p in face_landmarks]]
        face_contours = np.array(face_points).astype(np.single)
        for hand_landmarks in hands_landmarks:
            hand = [(point.x, point.y) for point in hand_landmarks.landmark]
            for finger in [4, 8, 20]:
                overlap = cv2.pointPolygonTest(face_contours, hand[finger], False)
                if overlap != -1:
                    return True
    return False

def get_avg_gaze(face):
    gaze_left = get_gaze(face, 476, 474, 263, 362)
    gaze_right = get_gaze(face, 471, 469, 33, 133)
    return round((gaze_left + gaze_right) / 2, 1)

def get_gaze(face, iris_L_side, iris_R_side, eye_L_corner, eye_R_corner):
    iris = (face[iris_L_side].x + face[iris_R_side].x, face[iris_L_side].y + face[iris_R_side].y)
    eye_center = (face[eye_L_corner].x + face[eye_R_corner].x, face[eye_L_corner].y + face[eye_R_corner].y)
    gaze_dist = dist.euclidean(iris, eye_center)
    eye_width = abs(face[eye_R_corner].x - face[eye_L_corner].x)
    gaze_relative = gaze_dist / eye_width if eye_width > 0 else 0
    if (eye_center[0] - iris[0]) < 0:
        gaze_relative *= -1
    return gaze_relative

def detect_gaze_change(avg_gaze, gaze_values):
    gaze_values.append(avg_gaze)
    if len(gaze_values) > MAX_FRAMES:
        gaze_values.pop(0)
    gaze_relative_matches = 1.0 * gaze_values.count(avg_gaze) / len(gaze_values)
    return "Change in gaze" if gaze_relative_matches < 0.01 else None

def get_lip_ratio(face):
    return get_aspect_ratio(face[0], face[17], face[61], face[291])

def get_area(image, topL, topR, bottomR, bottomL):
    topY = int((topR.y + topL.y) / 2 * image.shape[0])
    botY = int((bottomR.y + bottomL.y) / 2 * image.shape[0])
    leftX = int((topL.x + bottomL.x) / 2 * image.shape[1])
    rightX = int((topR.x + bottomR.x) / 2 * image.shape[1])
    return image[topY:botY, rightX:leftX]

def get_bpm_tells(cheekL, cheekR):
    global bpm_values, last_bpm_time
    cheekLwithoutBlue = np.average(cheekL[:, :, 1:3]) if cheekL.size > 0 else 0
    cheekRwithoutBlue = np.average(cheekR[:, :, 1:3]) if cheekR.size > 0 else 0
    signal.append(cheekLwithoutBlue + cheekRwithoutBlue)
    if len(signal) > 100 and time.time() - last_bpm_time > 5:
        fps = 30
        signal_np = np.array(signal[-100:])
        freqs = np.fft.fftfreq(len(signal_np)) * fps
        fft = np.abs(np.fft.fft(signal_np - np.mean(signal_np)))
        valid_idx = (freqs > 0.8) & (freqs < 3)
        if np.any(valid_idx):
            bpm = freqs[valid_idx][np.argmax(fft[valid_idx])] * 60
            bpm_values.append(bpm)
            signal[:] = signal[-50:]
            last_bpm_time = time.time()
    recent_avg_bpm = int(np.mean(bpm_values[-RECENT_FRAMES:])) if len(bpm_values) >= RECENT_FRAMES else 0
    bpm_display = f"BPM: {recent_avg_bpm}" if recent_avg_bpm > 0 else "BPM: ..."
    bpm_change = ""
    if len(bpm_values) > RECENT_FRAMES:
        all_avg_bpm = np.mean(bpm_values)
        avg_recent_bpm = np.mean(bpm_values[-RECENT_FRAMES:])
        bpm_delta = avg_recent_bpm - all_avg_bpm
        if bpm_delta > SIGNIFICANT_BPM_CHANGE:
            bpm_change = "Heart rate increasing"
        elif bpm_delta < -SIGNIFICANT_BPM_CHANGE:
            bpm_change = "Heart rate decreasing"
    return bpm_display, bpm_change

def add_text(image, tells, calibrated):
    text_y = 30
    if calibrated:
        for tell in tells.values():
            cv2.putText(image, tell['text'], (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, tell['text'], (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            text_y += 30

def add_truth_meter(image, tell_count):
    width = image.shape[1]
    sm = int(width / 64)
    bg = int(width / 3.2)
    # Simuler un "truth meter" avec un rectangle
    if tell_count:
        tellX = bg + int(bg / 4) * (tell_count - 1)
        cv2.rectangle(image, (tellX, int(0.9 * sm)), (tellX + int(sm / 2), int(2.1 * sm)), (0, 0, 255), 2)

# Boucle principale
blinks = [False] * MAX_FRAMES
hand_on_face = [False] * MAX_FRAMES
gaze_values = [0] * MAX_FRAMES
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_mesh.process(rgb_frame)
    hands_results = hands.process(rgb_frame)

    tells = decrement_tells(tells)
    lie_score = 0

    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        face = face_landmarks.landmark
        calibration_frames += 1
        calibrated = calibration_frames >= MAX_FRAMES

        # Repères faciaux
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # Pouls
        cheekL = get_area(frame, topL=face[449], topR=face[350], bottomR=face[429], bottomL=face[280])
        cheekR = get_area(frame, topL=face[121], topR=face[229], bottomR=face[50], bottomL=face[209])
        bpm_display, bpm_change = get_bpm_tells(cheekL, cheekR)
        tells['bpm'] = new_tell(bpm_display)
        if bpm_change:
            tells['bpm_change'] = new_tell(bpm_change)
            lie_score += 20

        # Clignements
        blinks.append(is_blinking(face))
        if len(blinks) > MAX_FRAMES:
            blinks.pop(0)
        blink_tell = get_blink_tell(blinks)
        if blink_tell:
            tells['blinking'] = new_tell(blink_tell)
            lie_score += 20

        # Mains sur visage
        hand_on_face.append(check_hand_on_face(hands_results.multi_hand_landmarks, face))
        if len(hand_on_face) > MAX_FRAMES:
            hand_on_face.pop(0)
        if hand_on_face[-1]:
            tells['hand'] = new_tell("Hand covering face")
            lie_score += 20

        # Regard
        avg_gaze = get_avg_gaze(face)
        gaze_tell = detect_gaze_change(avg_gaze, gaze_values)
        if gaze_tell:
            tells['gaze'] = new_tell(gaze_tell)
            lie_score += 20

        # Lèvres
        if get_lip_ratio(face) < LIP_COMPRESSION_RATIO:
            tells['lips'] = new_tell("Lip compression")
            lie_score += 20

        # Pourcentage
        lie_score = min(max(int(lie_score), 0), 80)
        lie_scores.append(lie_score)
        if len(lie_scores) > 10:
            lie_scores.pop(0)
        smoothed_lie_score = int(np.mean(lie_scores)) if lie_scores else lie_score
        cv2.putText(frame, f"{smoothed_lie_score}% menteur",
                    (450, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Messages gradués
        if smoothed_lie_score < 20:
            cv2.putText(frame, "Clean",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif smoothed_lie_score < 40:
            cv2.putText(frame, "Suspect",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif smoothed_lie_score < 60:
            cv2.putText(frame, "Lying",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Big Lie!",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Truth meter
        add_truth_meter(frame, len(tells))

    add_text(frame, tells, calibrated)

    # Optionnel : décommente pour enregistrer
    # out.write(frame)

    cv2.imshow("detecteur", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
face_mesh.close()
hands.close()