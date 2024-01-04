const character1Input = document.getElementById('character1');
const character2Input = document.getElementById('character2');
const generateBtn = document.getElementById('generateBtn');
const dialogueOutput = document.getElementById('dialogueOutput');
const modelSelect = document.getElementById('modelSelect'); // Model selection dropdown

let contextMaxLength = 512 ;
let logits = [];
let displayedText = "";
let modelPath = '';
let stoi = {};
let itos = {};
let model = null;
let currentSpeaker = character1Input.value; // Keep track of the current speaker
let bpeModel = false; // Flag to indicate if the current model is BPE
let generationInterval = null; // Global variable to hold the generation interval
let isGenerating = false; // Flag to indicate if text is being generated

// Initialize the model based on the selection
async function initModel() {

    let initialSpeakerAdded = false;

    if (isGenerating) {
        clearInterval(generationInterval);
        isGenerating = false;
    }
    if (modelSelect.value === "char") {displayedText = "begining\n";} else {displayedText = "<begin>";}
    dialogueOutput.innerHTML = '';

    if (modelSelect.value === 'bpe') {
        modelPath = 'ONNX_saved/bpe/exported_model.onnx';
        bpeModel = true;
        const list = ["<spkchg>", "something</w>", "different</w>", "including</w>", "because</w>", "through</w>", "ational</w>", "between</w>", "owever,</w>", "another</w>", "without</w>", "against</w>", "people</w>", "should</w>", "ations</w>", "before</w>", "you're</w>", "really</w>", "around</w>", "ation.</w>", "ation,</w>", "during</w>", "things</w>", "number</w>", "little</w>", "ation</w>", "about</w>", "their</w>", "which</w>", "would</w>", "other</w>", "there</w>", "don't</w>", "tions</w>", "could</w>", "first</w>", "after</w>", "thing</w>", "where</w>", "ating</w>", "right</w>", "aking</w>", "think</w>", "these</w>", "going</w>", "while</w>", "being</w>", "using</w>", "ement</w>", "might</w>", "still</w>", "ative</w>", "those</w>", "owing</w>", "ently</w>", "place</w>", "ready</w>", "tion.</w>", "years</w>", "great</w>", "There</w>", "until</w>", "three</w>", "under</w>", "every</w>", "since</w>", "start</w>", "uring</w>", "that</w>", "your</w>", "with</w>", "have</w>", "will</w>", "from</w>", "this</w>", "tion</w>", "they</w>", "ally</w>", "more</w>", "ting</w>", "like</w>", ".<begin>", "able</w>", "also</w>", "king</w>", "ment</w>", "ther</w>", "ated</w>", "This</w>", "ents</w>", "when</w>", "into</w>", "some</w>", "time</w>", "been</w>", "ding</w>", "what</w>", "make</w>", "them</w>", "just</w>", "than</w>", "were</w>", "ence</w>", "want</w>", "ing.</w>", "over</w>", "ing,</w>", "need</w>", "ance</w>", "most</w>", "very</w>", "only</w>", "said</w>", "sion</w>", "ning</w>", "ving</w>", "it's</w>", "work</w>", "then</w>", "know</w>", "sure</w>", "take</w>", "help</w>", "ough</w>", "good</w>", "at's</w>", "ound</w>", "used</w>", "tive</w>", "back</w>", "much</w>", "ates</w>", "such</w>", "many</w>", "even</w>", "It's</w>", "ying</w>", "ical</w>", "ties</w>", "down</w>", "port</w>", "side</w>", "each</w>", "find</w>", "come</w>", "ings</w>", "sing</w>", "last</w>", "What</w>", "well</w>", "look</w>", "ways</w>", "ited</w>", "self</w>", "same</w>", "dn't</w>", "made</w>", "When</w>", "They</w>", "akes</w>", "line</w>", "ower</w>", "less</w>", "on't</w>", "both</w>", "ends</w>", "best</w>", "next</w>", "ters</w>", "give</w>", "ular</w>", "lick</w>", "long</w>", "keep</w>", "part</w>", "ever</w>", "feel</w>", "year</w>", "here</w>", "ouse</w>", "ants</w>", "sive</w>", "form</w>", "lion</w>", "ause</w>", "ince</w>", "ould</w>", "fore</w>", "ttle</w>", "ween</w>", "the</w>", "and</w>", "ing</w>", "you</w>", "for</w>", "are</w>", "The</w>", "can</w>", "was</w>", "ent</w>", "ate</w>", "one</w>", "ers</w>", "all</w>", "not</w>", "has</w>", "but</w>", "ted</w>", "es.</w>", "ter</w>", "his</w>", "es,</w>", "ity</w>", "out</w>", "any</w>", "age</w>", "its</w>", "our</w>", "get</w>", "who</w>", "You</w>", "ght</w>", "est</w>", "ain</w>", "ure</w>", "ice</w>", "ely</w>", "ite</w>", "ect</w>", "ess</w>", "ard</w>", "new</w>", "end</w>", "ant</w>", "ase</w>", "day</w>", "had</w>", "her</w>", "der</w>", "old</w>", "use</w>", "own</w>", "ary</w>", "may</w>", "ame</w>", "how</w>", "ous</w>", "ast</w>", "er,</w>", "ome</w>", "er.</w>", "ver</w>", "see</w>", "ans</w>", "ed.</w>", "two</w>", "way</w>", "red</w>", "she</w>", "ong</w>", "le,</w>", "e's</w>", "ell</w>", "But</w>", "als</w>", "ace</w>", "ked</w>", "ber</w>", "ake</w>", "n't</w>", "ded</w>", "ust</w>", "ves</w>", "ed,</w>", "ble</w>", "ood</w>", "ine</w>", "ook</w>", "son</w>", "now</w>", "ack</w>", "ors</w>", "ile</w>", "ays</w>", "I'm</w>", "ave</w>", "For</w>", "try</w>", "<begin>", "ved</w>", "it.</w>", "him</w>", "ily</w>", "ill</w>", "ows</w>", "sts</w>", "off</w>", "tic</w>", "too</w>", "act</w>", "sed</w>", "ind</w>", "ore</w>", "ear</w>", "ven</w>", "ful</w>", "ool</w>", "ten</w>", "And</w>", "ese</w>", "ree</w>", "on,</w>", "ons</w>", "le.</w>", "set</w>", "'re</w>", "lic</w>", "few</w>", "on.</w>", "lot</w>", "ach</w>", "ick</w>", "man</w>", "ank</w>", "ond</w>", "top</w>", "did</w>", "ail</w>", "tor</w>", "put</w>", "les</w>", "develop", "en,</w>", "ass</w>", "say</w>", "ort</w>", "How</w>", "per</w>", "ade</w>", "ere</w>", "ink</w>", "t's</w>", "til</w>", "to</w>", "of</w>", "in</w>", "ed</w>", "es</w>", "is</w>", "on</w>", "al</w>", "er</w>", "an</w>", "it</w>", "or</w>", "as</w>", "be</w>", "s.</w>", "at</w>", "s,</w>", "e.</w>", "ly</w>", "e,</w>", "'s</w>", "by</w>", "ts</w>", "ve</w>", "en</w>", "se</w>", "le</w>", "y,</w>", "th</w>", "he</w>", "y.</w>", "up</w>", "if</w>", "ch</w>", "ad</w>", "ar</w>", "st</w>", "ow</w>", "et</w>", "If</w>", "so</w>", "ay</w>", "we</w>", "ic</w>", "do</w>", "ce</w>", "ds</w>", "t.</w>", ".:</w>", "ks</w>", "d.</w>", "de</w>", "am</w>", "my</w>", "t,</w>", "It</w>", "ll</w>", "ty</w>", "In</w>", "me</w>", "d,</w>", "ss</w>", "ge</w>", "ue</w>", "sh</w>", "us</w>", "el</w>", "om</w>", "no</w>", "go</w>", "a,</w>", "He</w>", "te</w>", "00</w>", "We</w>", "ey</w>", "ew</w>", "1.</w>", "ps</w>", "2.</w>", ").</w>", "ft</w>", "id</w>", "produc", "a.</w>", "'t</w>", "3.</w>", "ze</w>", "ry</w>", "Americ", "ms</w>", "4.</w>", "gn</w>", "5.</w>", "inform", "),</w>", "um</w>", "system", "ep</w>", "6.</w>", "As</w>", ".begin", "7.</w>", "re</w>", "govern", "8.</w>", "proble", "dy</w>", "import", "k,</w>", "ke</w>", "ut</w>", "ap</w>", "ld</w>", "person", "op</w>", "experi", "k.</w>", "dition", "ir</w>", "ul</w>", "differ", "owever", "a</w>", ".</w>", ",</w>", "s</w>", "y</w>", "t</w>", "e</w>", "I</w>", "d</w>", "k</w>", "o</w>", "l</w>", "p</w>", "n</w>", ")</w>", "m</w>", "-</w>", ":</w>", "?</w>", "!</w>", "g</w>", "A</w>", "inter", "0</w>", "r</w>", "h</w>", "x</w>", "f</w>", "5</w>", "ation", "2</w>", "'</w>", "chang", "i</w>", "1</w>", ";</w>", "w</w>", "S</w>", "speci", "3</w>", "4</w>", "provi", "6</w>", "under", "8</w>", "C</w>", "direc", "every", "consi", "7</w>", "possi", "appro", "gener", "struc", "examp", "stand", "progr", "incre", "busin", "inclu", "begin", "again", "devel", "gover", "</w>", "comp", "cont", "ther", "comm", "tion", "enti", "sion", "form", "pres", "ment", "coun", "stor", "high", "serv", "play", "poin", "year", "inst", "over", "requ", "you'", "poli", "them", "tran", "some", "work", "star", "resp", "part", "buil", "hand", "call", "cent", "curr", "chil", "ough", "Trum", "proc", "stud", "cour", "posi", "soci", "sear", "happ", "deci", "foll", "stre", "elec", "stat", "medi", "vers", "heal", "char", "rele", "grou", "lear", "plac", "mark", "writ", "your", "back", "issu", "port", "prof", "olog", "read", "with", "ener", "incl", "peri", "diff", "peop", "ever", "stem", "Amer", "con", "ent", "ter", "per", "pro", "ver", "all", "pre", "oun", "ain", "str", "for", "wor", "ing", "par", "end", "com", "min", "tic", "and", "ear", "est", "tim", "enc", "our", "rec", "ess", "ill", "ell", "off", "ail", "acc", "fin", "ali", "anc", "201", "mon", "lic", "app", "sec", "loc", "ast", "ang", "ati", "att", "cre", "sel", "por", "car", "der", "ori", "ach", "pri", "ass", "des", "tur", "fac", "ari", "gre", "man", "day", "ser", "lin", "own", "mar", "ani", "inv", "shi", "wat", "can", "mil", "cor", "fri", "oll", "rel", "ili", "you", "ack", "exp", "bas", "sur", "out", "ure", "sid", "tri", "wee", "sit", "ici", "Con", "fun", "sch", "amp", "sup", "ong", "dis", "cer", "pub", "cri", "loo", "mem", "mat", "war", "eas", "col", "arg", "emp", "cle", "ann", "the", "clo", "bre", "run", "any", "fam", "hel", "ful", "ant", "sub", "tre", "You", "leg", "bec", "dat", "tal", "dre", "lat", "ber", "ind", "ght", "sol", "ick", "lif", "fil", "ust", "int", "uni", "Mar", "Com", "mis", "lik", "ous", "pos", "sai", "reg", "fir", "tiv", "bet", "nec", "eng", "sul", "ist", "fic", "ide", "Pre", "gam", "gin", "ple", "dri", "mor", "air", "dec", "typ", "ven", "sti", "her", "get", "duc", "cas", "job", "nam", "let", "tor", "jec", "fer", "boo", "bor", "ash", "eff", "doc", "har", "cap", "aff", "sen", "new", "200", "sal", "Pro", "exc", "oul", "org", "foo", "pur", "lim", "dep", "vel", "son", "eci", "thr", "ble", "mer", "num", "tho", "cur", "bus", "kch", "whi", "som", "nex", "re", "in", "ar", "st", "en", "al", "er", "or", "li", "on", "an", "at", "ch", "th", "ad", "le", "as", "it", "di", "ac", "ic", "ri", "de", "ti", "es", "ro", "sh", "el", "un", "vi", "ag", "se", "em", "am", "si", "op", "ec", "om", "sp", "us", "tr", "ol", "ab", "ex", "ur", "ow", "ul", "lo", "ap", "ou", "ed", "et", "ne", "im", "qu", "pl", "te", "co", "su", "mo", "Th", "ir", "ci", "be", "il", "ai", "ut", "av", "ay", "to", "gr", "fi", "is", "oo", "sc", "um", "bo", "me", "do", "wh", "po", "ep", "bu", "no", "ha", "ni", "we", "dr", "cl", "pr", "br", "gi", "gh", "up", "oc", "aw", "he", "ak", "au", "Ch", "tu", "ph", "In", "bi", "ev", "fe", "mi", "St", "go", "tt", "wi", "so", "hi", "19", "fl", "ei", "pe", "fr", "Wh", "bl", "ef", "af", "gu", "ub", "ob", "gn", "sm", "cr", "sk", "ho", "du", "gg", "uc", "ge", "An", "ve", "id", "kn", "Al", "pp", "Re", "20", "ew", "ff", "I'", "En", "mp", "ud", "vo", "la", "e-", "ra", "10", "ea", "jo", "pi", "On", "Tr", "ma", "No", "Un", "ke", "Ac", "of", "ey", "Mo", "sl", "Ar", "Jo", "Ad", "eg", "wr", "tw", "Di", "sw", "Fr", "Li", "Po", ",0", "sy", "yp", "og", "mu", "t", "s", "c", "e", "p", "d", "i", "f", "b", "m", "S", "l", "h", "u", "o", "r", "g", "-", "n", "a", "y", "M", "C", "w", "v", "A", "B", "k", "D", "P", "T", "1", "R", "H", "N", "E", "F", "G", "W", "O", "L", "z", ".", "(", "I", "U", "2", "j", "J", "K", "x", "3", "V", "'", "0", "9", "5", "4", "Y", "8", "6", "7", "!", "X", "Q", ":", "Z", ",", ")", "?", "q", ";", "<unk>"];
        list.forEach((item, index) => {
             stoi[item] = index;
            });
            
// Create a dictionary (object) mapping integers to strings

        list.forEach((item, index) => {
            itos[index] = item;
        });
        contextMaxLength = 256;

        }
        else {
        modelPath = 'ONNX_saved/char/exported_model.onnx';
        // Character to Integer mapping (based on your Python stoi)
        stoi = {'\n': 0, ' ': 1, '!': 2, '"': 3, "'": 4, '(': 5, ')': 6, ',': 7, '-': 8, '.': 9, '0': 10, '1': 11, '2': 12, '3': 13, '4': 14, '5': 15, '6': 16, '7': 17, '8': 18, '9': 19, ':': 20, ';': 21, '?': 22, '[': 23, ']': 24, 'a': 25, 'b': 26, 'c': 27, 'd': 28, 'e': 29, 'f': 30, 'g': 31, 'h': 32, 'i': 33, 'j': 34, 'k': 35, 'l': 36, 'm': 37, 'n': 38, 'o': 39, 'p': 40, 'q': 41, 'r': 42, 's': 43, 't': 44, 'u': 45, 'v': 46, 'w': 47, 'x': 48, 'y': 49, 'z': 50, '{': 51, '}': 52};
        itos = {0: '\n', 1: ' ', 2: '!', 3: '"', 4: "'", 5: '(', 6: ')', 7: ',', 8: '-', 9: '.', 10: '0', 11: '1', 12: '2', 13: '3', 14: '4', 15: '5', 16: '6', 17: '7', 18: '8', 19: '9', 20: ':', 21: ';', 22: '?', 23: '[', 24: ']', 25: 'a', 26: 'b', 27: 'c', 28: 'd', 29: 'e', 30: 'f', 31: 'g', 32: 'h', 33: 'i', 34: 'j', 35: 'k', 36: 'l', 37: 'm', 38: 'n', 39: 'o', 40: 'p', 41: 'q', 42: 'r', 43: 's', 44: 't', 45: 'u', 46: 'v', 47: 'w', 48: 'x', 49: 'y', 50: 'z', 51: '{', 52: '}'};

        }
    console.log("Loading model...");
    model = await ort.InferenceSession.create(modelPath, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
    console.log("Model loaded");
}






function formatText(text) {
    let newSentence = true;
    let formattedText = '';

    for (let char of text) {
        if (newSentence && char.match(/[a-z]/i)) {
            // Mettre le caractère en majuscule s'il débute une nouvelle phrase
            formattedText += char.toUpperCase();
            newSentence = false;
        } else if (char === '.') {
            // Si le caractère est un point, la prochaine lettre doit être en majuscule
            formattedText += char;
            newSentence = true;
        } else {
            // Pour les 'i' isolés, les convertir en majuscules
            if (char === 'i' && (formattedText.endsWith(' ') || formattedText === '')) {
                formattedText += 'I';
            } else {
                formattedText += char;
            }
        }
    }

    return formattedText;
}



function replacePlaceholdersAndHandleSpeakers(text) {
    if (bpeModel) {
        // Convert </w> to spaces and handle speaker changes and new lines for sentences
        text = text.replace(/<\/w>/g, ' ');

        // Handling speaker changes and new lines for sentences
        let formattedText = '';
        let lines = text.split('\n');

        lines.forEach((line, index) => {
            // Add speaker name for each line
            console.log(currentSpeaker)
            if (initialSpeakerAdded){
            formattedText += `<span class="char${currentSpeaker === character1Input.value ? '1' : '2'}">${currentSpeaker}</span>: `;
            }
            // Add the line text
            formattedText += line;

            // Check if the line ends with a punctuation mark
            if (!line.endsWith('.') && !line.endsWith('!') && !line.endsWith('?')) {
                formattedText += '.';
            }

            // Add a new line break
            formattedText += '<br>';
        });

        return formattedText;
    } else {
        // For character-level model, replace {1} and {2} with character names
        return text.replace(/\{1\}/g, `<span class="char1">${character1Input.value}</span>`)
                   .replace(/\{2\}/g, `<span class="char2">${character2Input.value}</span>`);
    }
}


// Function to convert a string to an array of integers
function stringToIntArray(str) {
    const intArray = [];
    console.log("str", str)
    for (let i = 0; i < str.length; i++) {
        if (bpeModel == true && i == 0)
        {i = 7 ;
         intArray.push(stoi["<begin>"])
        }
        else {
        const char = str[i];
        intArray.push(stoi[char]);
        if (stoi[char] === undefined) {
            console.error("Character not in vocabulary:", char);
        }
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

    // Model inference
    const output = await model.run({ 'input': inputTensor });

    if (bpeModel) {logits = output['output'].data.slice(-1070);} else {logits = output['output'].data.slice(-53);}


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
    let speakerChange = false; // Flag to indicate a speaker change

    for (let i = 0; i < cdf.length; i++) {
        if (r < cdf[i]) {
            nextCharId = i;
            break;
        }
    }
    if (bpeModel && itos[nextCharId] === '<spkchg>') {
        speakerChange = true; // Set the flag for a speaker change
    }
    // console.log("Next character id:");
    // console.log(nextCharId);
    return { nextCharId, speakerChange };

    // output_data = softmax(output_data)

    // let nextCharId = multinomial(output_data)
    
    // console.log(itos[nextCharId], nextCharId)
    // // console.log("Next character id:");
    // // console.log(nextCharId);
    // return nextCharId;
    
    // //return nextToken;
}

async function displayGeneratedText() {
    let context = stringToIntArray(displayedText);
    displayedText = "";

    // Define the initial speaker
    let initialSpeaker = character1Input.value;
    let currentSpeaker = initialSpeaker;

    // Flag to track if the initial speaker's name has been added
    initialSpeakerAdded = false;

    async function appendNextChar() {
        if (context.length >= contextMaxLength) {
            context.shift();
        }

        const { nextCharId, speakerChange } = await generateText(context);
        context.push(nextCharId);

        // Add the initial speaker's name for the first line (if not added yet)

        if ((!initialSpeakerAdded)&&(bpeModel)) {
            displayedText += `<span class="char${initialSpeaker === character1Input.value ? '1' : '2'}">${initialSpeaker}</span>: `;
            initialSpeakerAdded = true;
        }

        if (speakerChange) {
            // Change the current speaker
            currentSpeaker = (currentSpeaker === character1Input.value) ? character2Input.value : character1Input.value;
            // Add a line break and the new speaker's name
            displayedText += '<br><span class="char' + (currentSpeaker === character1Input.value ? '1' : '2') + '">' + currentSpeaker + '</span>: ';
        } else {
            const nextChar = itos[nextCharId];
            displayedText += nextChar;
        }
        if (!bpeModel){
        displayedText = formatText(displayedText);
        }
        const formattedText = replacePlaceholdersAndHandleSpeakers(displayedText).replace(/\n/g, '<br>');
        dialogueOutput.innerHTML = formattedText;
    }

    generationInterval = setInterval(appendNextChar, 100);
    isGenerating = true;
}




generateBtn.addEventListener('click', () => {
    displayGeneratedText();
});
modelSelect.addEventListener('change', () => {
    initModel(); // Re-initialize the model when selection changes
});
initModel();
