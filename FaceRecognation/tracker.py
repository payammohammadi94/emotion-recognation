import numpy as np


class FaceTracker:
    def __init__(self, emotion_label, same_threshold=0.5, new_threshold=0.7):
        self.emotion_label = emotion_label
        self.same_threshold = same_threshold
        self.new_threshold = new_threshold

    def update(self, face_id_dict, face_embed, face_emotion, time_stamp):
        if len(face_id_dict) == 0:
            face_id_dict[0] = {
                "face_embed" : face_embed,
                "mean_emotion_probs" : face_emotion,
                "emotion_probs" : [face_emotion],
                "time_stamp" : [time_stamp],
                "n_record" : 1,
            }
            return face_id_dict, 0

        compare_flag, closest_id, min_distance = self.compare_id(face_id_dict, face_embed)

        if compare_flag >= 0:
            id_n_record = face_id_dict[compare_flag]["n_record"]
            id_face_embed = face_id_dict[compare_flag]["face_embed"]
            id_face_emotion = face_id_dict[compare_flag]["emotion_probs"]
            id_time_stamp = face_id_dict[compare_flag]["time_stamp"]

            id_face_embed = (id_face_embed * id_n_record + face_embed)  / (id_n_record + 1)
            id_face_emotion.append(face_emotion)
            id_mean_face_emotion = np.stack(id_face_emotion, axis=0).mean(axis=0)
            id_time_stamp.append(time_stamp)
            id_n_record = id_n_record + 1
            

            face_id_dict[compare_flag] = {
                "face_embed" : id_face_embed,
                "mean_emotion_probs" : id_mean_face_emotion,
                "emotion_probs" : id_face_emotion,
                "time_stamp" : id_time_stamp,
                "n_record" : id_n_record,
            }
            return face_id_dict, compare_flag
        elif compare_flag == -1:
            new_id = len(face_id_dict)
            face_id_dict[new_id] = {
                "face_embed" : face_embed,
                "mean_emotion_probs" : face_emotion,
                "emotion_probs" : [face_emotion],
                "time_stamp" : [time_stamp],
                "n_record" : 1,
            }
            return face_id_dict, new_id
        else:
            # Uncertain state: بین threshold ها - استفاده از نزدیک‌ترین ID با وزن کمتر
            if closest_id is not None:
                id_n_record = face_id_dict[closest_id]["n_record"]
                id_face_embed = face_id_dict[closest_id]["face_embed"]
                id_face_emotion = face_id_dict[closest_id]["emotion_probs"]
                id_time_stamp = face_id_dict[closest_id]["time_stamp"]

                # استفاده از وزن کمتر برای update (0.3 به جای 1.0)
                update_weight = 0.3
                id_face_embed = (id_face_embed * id_n_record + face_embed * update_weight) / (id_n_record + update_weight)
                id_face_emotion.append(face_emotion)
                id_mean_face_emotion = np.stack(id_face_emotion, axis=0).mean(axis=0)
                id_time_stamp.append(time_stamp)
                id_n_record = id_n_record + 1

                face_id_dict[closest_id] = {
                    "face_embed" : id_face_embed,
                    "mean_emotion_probs" : id_mean_face_emotion,
                    "emotion_probs" : id_face_emotion,
                    "time_stamp" : id_time_stamp,
                    "n_record" : id_n_record,
                }
                return face_id_dict, closest_id
            else:
                # اگر هیچ ID نزدیکی پیدا نشد، ID جدید ایجاد کن
                new_id = len(face_id_dict)
                face_id_dict[new_id] = {
                    "face_embed" : face_embed,
                    "mean_emotion_probs" : face_emotion,
                    "emotion_probs" : [face_emotion],
                    "time_stamp" : [time_stamp],
                    "n_record" : 1,
                }
                return face_id_dict, new_id
    

    def compare_id(self, face_id_dict, face_embed):
        all_embed = [face_id_dict[id]["face_embed"] for id in range(len(face_id_dict))]
        all_embed = np.concatenate(all_embed, axis=0)

        distance = self.euclidean_distance(self.normalize(face_embed), self.normalize(all_embed))
        min_id, min_distance = np.argmin(distance), np.min(distance)

        if min_distance <= self.same_threshold:
            return min_id, min_id, min_distance
        elif min_distance >= self.new_threshold:
            return -1, None, min_distance
        else:
            # Uncertain state: بین threshold ها
            return -2, min_id, min_distance
    
    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))
    
    def normalize(self, x):
        norm = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))
        return x / (norm + 1e-8)
