from flask import Flask, request, render_template
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import MT5Config

app = Flask(__name__)


tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small', legacy=False)


config = MT5Config()

config.decoder_start_token_id = config.pad_token_id


model = MT5ForConditionalGeneration(config)


LANG_TOKEN_MAPPING = {
    'auto': '',
    'en': '<en>',
    'vi': '<vi>',
    'es': '<es>',

}

special_tokens_dict = {'additional_special_tokens': list(LANG_TOKEN_MAPPING.values())}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def load_model(model, checkpoint_path, device):
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])


load_model(model, "D:\dự án công nghệ thông tin\_epoch_7.pt\_epoch_7.pt", device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    text_to_translate = request.form['text']
    input_lang = request.form.get('input_language')  
    target_lang = request.form['language']

    if input_lang == target_lang:
        translation = "Input and output languages must be different."
    else:
        
        prefix = LANG_TOKEN_MAPPING.get(target_lang, '')
        if input_lang != 'auto':
            text_to_translate = LANG_TOKEN_MAPPING.get(input_lang, '') + " " + text_to_translate
        text_to_translate_with_prefix = f"{prefix} {text_to_translate}"

        translation = translate_sentences(model, [text_to_translate_with_prefix], device, tokenizer)[0]

    return render_template('index.html', translation=translation)

def translate_sentences(model, sentences, device, tokenizer, max_length=512):
    model.eval()
    translations = []

    with torch.no_grad():
        for sentence in sentences:
            input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
            outputs = model.generate(input_ids, max_length=max_length)
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translations.append(translation)

    return translations

if __name__ == '__main__':
    app.run(debug=True)
