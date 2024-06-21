from .singing_transcription import SingingTranscription
def get_note_JDC(audio):
    ST = SingingTranscription()

    """ load model """
    model_ST = ST.load_model("../data/weight_ST.hdf5", TF_summary=False)
    fl_note = ST.predict_melody(model_ST, audio)
    return fl_note
