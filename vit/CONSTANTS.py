DIR_COCO_IMAGE_FOLDER="../coco/test2017/"
DIR_COCO_JSON_FILE="../coco/annotations/image_info_test2017.json"
POST_PROCESS_THRESH=0.25
BBOX_TOPK=100
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
    "potted plant", "bed", "dining table", "toilet", "TV", "laptop", "mouse", 
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
    "toothbrush"
]

# released by facebook research official
DETR_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

DETR_DICT = {
    "0": "N/A",
    "1": "person",
    "2": "bicycle",
    "3": "car",
    "4": "motorcycle",
    "5": "airplane",
    "6": "bus",
    "7": "train",
    "8": "truck",
    "9": "boat",
    "10": "traffic light",
    "11": "fire hydrant",
    "12": "street sign",
    "13": "stop sign",
    "14": "parking meter",
    "15": "bench",
    "16": "bird",
    "17": "cat",
    "18": "dog",
    "19": "horse",
    "20": "sheep",
    "21": "cow",
    "22": "elephant",
    "23": "bear",
    "24": "zebra",
    "25": "giraffe",
    "26": "hat",
    "27": "backpack",
    "28": "umbrella",
    "29": "shoe",
    "30": "eye glasses",
    "31": "handbag",
    "32": "tie",
    "33": "suitcase",
    "34": "frisbee",
    "35": "skis",
    "36": "snowboard",
    "37": "sports ball",
    "38": "kite",
    "39": "baseball bat",
    "40": "baseball glove",
    "41": "skateboard",
    "42": "surfboard",
    "43": "tennis racket",
    "44": "bottle",
    "45": "plate",
    "46": "wine glass",
    "47": "cup",
    "48": "fork",
    "49": "knife",
    "50": "spoon",
    "51": "bowl",
    "52": "banana",
    "53": "apple",
    "54": "sandwich",
    "55": "orange",
    "56": "broccoli",
    "57": "carrot",
    "58": "hot dog",
    "59": "pizza",
    "60": "donut",
    "61": "cake",
    "62": "chair",
    "63": "couch",
    "64": "potted plant",
    "65": "bed",
    "66": "mirror",
    "67": "dining table",
    "68": "window",
    "69": "desk",
    "70": "toilet",
    "71": "door",
    "72": "tv",
    "73": "laptop",
    "74": "mouse",
    "75": "remote",
    "76": "keyboard",
    "77": "cell phone",
    "78": "microwave",
    "79": "oven",
    "80": "toaster",
    "81": "sink",
    "82": "refrigerator",
    "83": "blender",
    "84": "book",
    "85": "clock",
    "86": "vase",
    "87": "scissors",
    "88": "teddy bear",
    "89": "hair drier",
    "90": "toothbrush"
  }

DETR_DICT_LABEL2ID = {
    "N/A": 0,
    "airplane": 5,
    "apple": 53,
    "backpack": 27,
    "banana": 52,
    "baseball bat": 39,
    "baseball glove": 40,
    "bear": 23,
    "bed": 65,
    "bench": 15,
    "bicycle": 2,
    "bird": 16,
    "blender": 83,
    "boat": 9,
    "book": 84,
    "bottle": 44,
    "bowl": 51,
    "broccoli": 56,
    "bus": 6,
    "cake": 61,
    "car": 3,
    "carrot": 57,
    "cat": 17,
    "cell phone": 77,
    "chair": 62,
    "clock": 85,
    "couch": 63,
    "cow": 21,
    "cup": 47,
    "desk": 69,
    "dining table": 67,
    "dog": 18,
    "donut": 60,
    "door": 71,
    "elephant": 22,
    "eye glasses": 30,
    "fire hydrant": 11,
    "fork": 48,
    "frisbee": 34,
    "giraffe": 25,
    "hair drier": 89,
    "handbag": 31,
    "hat": 26,
    "horse": 19,
    "hot dog": 58,
    "keyboard": 76,
    "kite": 38,
    "knife": 49,
    "laptop": 73,
    "microwave": 78,
    "mirror": 66,
    "motorcycle": 4,
    "mouse": 74,
    "orange": 55,
    "oven": 79,
    "parking meter": 14,
    "person": 1,
    "pizza": 59,
    "plate": 45,
    "potted plant": 64,
    "refrigerator": 82,
    "remote": 75,
    "sandwich": 54,
    "scissors": 87,
    "sheep": 20,
    "shoe": 29,
    "sink": 81,
    "skateboard": 41,
    "skis": 35,
    "snowboard": 36,
    "spoon": 50,
    "sports ball": 37,
    "stop sign": 13,
    "street sign": 12,
    "suitcase": 33,
    "surfboard": 42,
    "teddy bear": 88,
    "tennis racket": 43,
    "tie": 32,
    "toaster": 80,
    "toilet": 70,
    "toothbrush": 90,
    "traffic light": 10,
    "train": 7,
    "truck": 8,
    "tv": 72,
    "umbrella": 28,
    "vase": 86,
    "window": 68,
    "wine glass": 46,
    "zebra": 24
  }
VAL_SUBSET_SIZE=100
MAX_COUNT = 100

BRID_FAMILY_NAME_LATIN = ["Struthionidae","Rheidae","Apterygidae","Casuariidae","Tinamidae","Anhimidae","Anseranatidae","Anatidae","Megapodiidae","Cracidae","Numididae","Odontophoridae","Phasianidae","Podargidae","Steatornithidae","Nyctibiidae","Caprimulgidae","Aegothelidae","Hemiprocnidae","Apodidae","Trochilidae","Musophagidae","Otididae","Cuculidae","Mesitornithidae","Pteroclidae","Columbidae","Heliornithidae","Sarothruridae","Rallidae","Psophiidae","Gruidae","Aramidae","Podicipedidae","Phoenicopteridae","Turnicidae","Burhinidae","Chionidae","Pluvianellidae","Haematopodidae","Ibidorhynchidae","Recurvirostridae","Charadriidae","Pluvianidae","Rostratulidae","Jacanidae","Pedionomidae","Thinocoridae","Scolopacidae","Dromadidae","Glareolidae","Laridae","Stercorariidae","Alcidae","Rhynochetidae","Eurypygidae","Phaethontidae","Gaviidae","Spheniscidae","Oceanitidae","Diomedeidae","Hydrobatidae","Procellariidae","Ciconiidae","Fregatidae","Sulidae","Anhingidae","Phalacrocoracidae","Threskiornithidae","Ardeidae","Scopidae","Balaenicipitidae","Pelecanidae","Opisthocomidae","Cathartidae","Sagittariidae","Pandionidae","Accipitridae","Tytonidae","Strigidae","Coliidae","Leptosomidae","Trogonidae","Upupidae","Phoeniculidae","Bucorvidae","Bucerotidae","Coraciidae","Brachypteraciidae","Alcedinidae","Todidae","Momotidae","Meropidae","Galbulidae","Bucconidae","Capitonidae","Semnornithidae","Ramphastidae","Megalaimidae","Lybiidae","Indicatoridae","Picidae","Cariamidae","Falconidae","Strigopidae","Cacatuidae","Psittacidae","Psittaculidae","Acanthisittidae","Sapayoidae","Philepittidae","Eurylaimidae","Calyptomenidae","Pittidae","Furnariidae","Thamnophilidae","Formicariidae","Grallariidae","Conopophagidae","Rhinocryptidae","Melanopareiidae","Tyrannidae","Cotingidae","Pipridae","Tityridae","Menuridae","Atrichornithidae","Ptilonorhynchidae","Climacteridae","Maluridae","Meliphagidae","Dasyornithidae","Pardalotidae","Acanthizidae","Pomatostomidae","Orthonychidae","Cnemophilidae","Melanocharitidae","Paramythiidae","Callaeidae","Notiomystidae","Psophodidae","Cinclosomatidae","Platysteiridae","Malaconotidae","Machaerirhynchidae","Vangidae","Pityriasidae","Artamidae","Rhagologidae","Aegithinidae","Campephagidae","Mohouidae","Neosittidae","Eulacestomatidae","Oreoicidae","Falcunculidae","Pachycephalidae","Vireonidae","Oriolidae","Dicruridae","Rhipiduridae","Monarchidae","Platylophidae","Laniidae","Corvidae","Corcoracidae","Melampittidae","Ifritidae","Paradisaeidae","Petroicidae","Picathartidae","Chaetopidae","Eupetidae","Bombycillidae","Ptiliogonatidae","Hypocoliidae","Dulidae","Mohoidae","Hylocitreidae","Stenostiridae","Paridae","Remizidae","Nicatoridae","Panuridae","Alaudidae","Pycnonotidae","Hirundinidae","Pnoepygidae","Macrosphenidae","Cettiidae","Scotocercidae","Erythrocercidae","Hyliidae","Aegithalidae","Phylloscopidae","Acrocephalidae","Locustellidae","Donacobiidae","Bernieridae","Cisticolidae","Sylviidae","Paradoxornithidae","Zosteropidae","Timaliidae","Pellorneidae","Alcippeidae","Leiothrichidae","Modulatricidae","Promeropidae","Irenidae","Regulidae","Elachuridae","Hyliotidae","Troglodytidae","Polioptilidae","Sittidae","Tichodromidae","Certhiidae","Salpornithidae","Mimidae","Sturnidae","Buphagidae","Turdidae","Muscicapidae","Cinclidae","Chloropseidae","Dicaeidae","Nectariniidae","Passeridae","Ploceidae","Estrildidae","Viduidae","Peucedramidae","Prunellidae","Motacillidae","Urocynchramidae","Fringillidae","Calcariidae","Rhodinocichlidae","Emberizidae","Passerellidae","Calyptophilidae","Phaenicophilidae","Nesospingidae","Spindalidae","Zeledoniidae","Teretistridae","Icteriidae","Icteridae","Parulidae","Mitrospingidae","Cardinalidae","Thraupidae"]

def GET_PROMPT(caption_text, probs, labels):
    prob_0 = str(probs[0][0].item())
    prob_1 = str(probs[0][1].item())
    prob_2 = str(probs[0][2].item())
    label_0, label_1, label_2 = labels
    PROMPT = f"""You are an expert ornithologist with extensive experience in bird identification and research. You have been provided with complementary AI model outputs to help identify and describe a bird in an image:\n

    1. Image Caption Model Output:
    {caption_text}

    2. Bird Classification Results:
    - Top prediction: {label_0} (Confidence: {prob_0})
    - Second prediction: {label_1} (Confidence: {prob_1})
    - Third prediction: {label_2} (Confidence: {prob_2})

    Based on these observations, please:
    1. Determine the most likely species
    2. Explain why this identification makes sense given both the visual description and classification results
    3. If there are any discrepancies between the caption and classifications, explain what might cause this
    4. Provide brief information about the identified species' typical behavior and habitat

    If you're uncertain about the exact species, please explain why and discuss the most likely possibilities.
    """
    return PROMPT
  
def GET_TRANSLATE_PROMPT(original_language):
    PROMPT = f"Please translate the following text from English to German:\n\n{original_language}"
    return PROMPT