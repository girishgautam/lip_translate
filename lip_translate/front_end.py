from api import *
def front_end(loaded_data):
    success = True
    i = 0
    frames = []

    while success:

        loaded_frame = loaded_data['arr_0'][i].tolist()
        # print(len(loaded_frame), len((loaded_frame)[0]))
        if loaded_frame is not None:
            success = True
            # print(success)
            frames.append(loaded_frame)
            i += 1
            if i % 5 == 0:
                print(len(frames))
                # response = requests.post("http://127.0.0.1:8000/predict", json=json.dumps(frames))
                print(test_function(frames))
                    # print(response)
                # if response.ok:
                #     frames = []
                # else:
                #     print(response)


front_end(np.load('/home/girishj/code/girishgautam/lip_translate/mathilda_test.npz'))
