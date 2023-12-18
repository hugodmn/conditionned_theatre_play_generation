
const character1Input = document.getElementById('character1');
const character2Input = document.getElementById('character2');
const generateBtn = document.getElementById('generateBtn');
const dialogueOutput = document.getElementById('dialogueOutput');

const contextMaxLength = 512;
 // Starting text
//pad displayedText to contextMaxLength

// for (let i = 0; i < contextMaxLength-"begining\n".length; i++) {
//     displayedText += ' ';
// }
let displayedText = "begining\n";

const title = document.createElement('h2');
title.textContent = "Character Dialogue";
dialogueOutput.before(title);


// Model related variables
const modelPath = 'ONNX_saved/exported_model.onnx';
let model = null;


// Character to Integer mapping (based on your Python stoi)
const stoi = {'\n': 0, ' ': 1, '!': 2, '"': 3, "'": 4, '(': 5, ')': 6, ',': 7, '-': 8, '.': 9, '0': 10, '1': 11, '2': 12, '3': 13, '4': 14, '5': 15, '6': 16, '7': 17, '8': 18, '9': 19, ':': 20, ';': 21, '?': 22, '[': 23, ']': 24, 'a': 25, 'b': 26, 'c': 27, 'd': 28, 'e': 29, 'f': 30, 'g': 31, 'h': 32, 'i': 33, 'j': 34, 'k': 35, 'l': 36, 'm': 37, 'n': 38, 'o': 39, 'p': 40, 'q': 41, 'r': 42, 's': 43, 't': 44, 'u': 45, 'v': 46, 'w': 47, 'x': 48, 'y': 49, 'z': 50, '{': 51, '}': 52};

const itos = {0: '\n', 1: ' ', 2: '!', 3: '"', 4: "'", 5: '(', 6: ')', 7: ',', 8: '-', 9: '.', 10: '0', 11: '1', 12: '2', 13: '3', 14: '4', 15: '5', 16: '6', 17: '7', 18: '8', 19: '9', 20: ':', 21: ';', 22: '?', 23: '[', 24: ']', 25: 'a', 26: 'b', 27: 'c', 28: 'd', 29: 'e', 30: 'f', 31: 'g', 32: 'h', 33: 'i', 34: 'j', 35: 'k', 36: 'l', 37: 'm', 38: 'n', 39: 'o', 40: 'p', 41: 'q', 42: 'r', 43: 's', 44: 't', 45: 'u', 46: 'v', 47: 'w', 48: 'x', 49: 'y', 50: 'z', 51: '{', 52: '}'};

// Initialize the model
async function initModel() {
    console.log("Loading model...");
    model = await ort.InferenceSession.create(modelPath, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    console.log("Model loaded");
}

// Function to replace placeholders with character names
function replacePlaceholders(text, char1, char2) {
    return text.replace(/\{1\}/g, `<span class="char1">${char1}</span>`)
               .replace(/\{2\}/g, `<span class="char2">${char2}</span>`);
}


// Function to convert a string to an array of integers
function stringToIntArray(str) {
    const intArray = [];
    console.log("str", str)
    for (let i = 0; i < str.length; i++) {
        const char = str[i];
        intArray.push(stoi[char]);

        if (stoi[char] === undefined) {
            console.error("Character not in vocabulary:", char);
        }
       
    }
    return intArray;
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
}

// Multinomial sampling function
function multinomial(probs) {
    const r = Math.random();
    let cumSum = 0;
    for (let i = 0; i < probs.length; i++) {
        cumSum += probs[i];
        if (r < cumSum) {
            return i;
        }
    }
    return probs.length - 1;
}

async function generateText(input) {


    //console.log("input", input)
    const inputTensor = new ort.Tensor('int32', new Int32Array(input), [1, input.length]);
    console.log("inputTensor", inputTensor)
    // Model inference
    const output = await model.run({ 'input': inputTensor });


    let logits = output['output'].data.slice(-53);


    // const probs = softmax(logits);

    // // Sample the next token using multinomial sampling
    // const nextToken = multinomial(probs);
    let output_data = logits
   // const output_data = output.output.data;

    const sum = output_data.reduce((a, b) => a + Math.exp(b), 0);
    const normalized = output_data.map(x => Math.exp(x) / sum);
    

    //! Sampling from the distribution
    // Cumulative distribution function
    // console.log("CDF");
    const cdf = [];
    let sum2 = 0;
    for (let i = 0; i < normalized.length; i++) {
        sum2 += normalized[i];
        cdf.push(sum2);
    }

    // Sample from the CDF

    const r = Math.random();
    // console.log("r:");
    // console.log(r);

    let nextCharId = 0;
    for (let i = 0; i < cdf.length; i++) {
        if (r < cdf[i]) {
            nextCharId = i;
            break;
        }
    }

    // console.log("Next character id:");
    // console.log(nextCharId);
    return nextCharId;


    // output_data = softmax(output_data)

    // let nextCharId = multinomial(output_data)
    
    // console.log(itos[nextCharId], nextCharId)
    // // console.log("Next character id:");
    // // console.log(nextCharId);
    // return nextCharId;
    
    // //return nextToken;
}


async function displayGeneratedText() {
    let context = stringToIntArray(displayedText); // Convert displayed text to integer array
    displayedText = " ";
    async function appendNextChar() {
        if (context.length >= contextMaxLength) {
            context.shift(); // Maintain context size
        }

        const nextCharId = await generateText(context);
        context.push(nextCharId);

        const nextChar = itos[nextCharId]; // Decode the next character
        displayedText += nextChar; // Append the next character to the displayed text
        const formattedText = replacePlaceholders(displayedText, character1Input.value, character2Input.value).replace(/\n/g, '<br>');
        dialogueOutput.innerHTML = formattedText;
    }

    const generationInterval = setInterval(appendNextChar, 100);
    // Optional: clearInterval(generationInterval);
}

generateBtn.addEventListener('click', () => {
    displayGeneratedText();
});

//initModel();
