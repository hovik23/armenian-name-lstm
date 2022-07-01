
# Armenian Name LSTM Generator
This project will be described in both Armenian and English languages. It is implemented to create some new names similar to Armenian ethnic names using **Deep Learning**. The [database](https://en.wiktionary.org/wiki/Category:Armenian_given_names) consists of 422 Armenian names written in Armenian language. Unfortunately, the given size of the dataset is quite small and having more names would definitely improve the result, though the result was fine. Also, in this exact situation it's also important to keep the dataset clear to avoid some weird or extremely unusual names that could make the result much worse. So, keeping all these factors in mind, a small ~~but gold~~ dataset was used.

Այս պրոյեկտը նկարագրվելու է հայերեն և անգլերեն լեզուներով։ Սա իրականացված է նոր հայկական անուններ մտածելու համար, որոնք նման կլինեն էթնիկ հայկական անուններին, օգտագործելով Deep Learning։ Տվյալները պարունակում են 422 անուն գրած հայերեն լեզվով։ Ցավոք, տվյալների քանակը բավականին քիչ է և ունենալով ավելի շատ անուններ արցունքը բնականաբար ավելի լավ կստացվի, բայց ստացված արցունքն էլ բավականին լավն էր։ Իդեպ, այս կոնկրետ խնդրում կարևոր է թվյալները պահել մաքուր, այսինքն պետք չի օգտագործել տարորինակ և շատ հազվադեպ օգտագործվող անուններ, որովհետեւ դա կարող է փչացնի արդյունքը։ Այսպես, հաշվի աենելով բոլոր թվարկված ֆակտորները, պոքր բայց լավ տվյալների բազա էր օգտագործված։

## Prerequisite | Նախադրյալներ
* PyTorch;
* Python.

## Idea | Գաղափար
The idea behind this is generating some new names using Deep Learning. It's clear that we're going to deal with sequences of letters because it's impossible to create new words using words. So, we can define that the problem is a **character-level** problem. So, keeping it simple - as the idea of this project is using sequences, it becomes clear that a **RNN** (Recurrent Neural Network) should be used. Also as we can't clearly provide any $\bm{X}$ and $\bm{y}$ (i.e. *data* and *target* respectively) it's going to be a generator which will generate a letter by letter but also trying to take into account previous printed letters.
Basically, generating a name is not a task where a short term memory problem (*Short Term Memory Loss*) could be critical because a usual name is probably never that long. It means a simple RNN could be used. But actually there is one more problem. The main thing that leads to short term memory problem is the vanishing gradient problem. As you probably know, training neural net means nothing more but adjusting its values. Neural net tries make a prediction and then it compares what it has done and the actual true value so it could try be better next time. And the gradient, which is being calculated by the net itself during the training, shows *how much* the net should adjust its values during back-propagation. And here comes the *Vanishing Gradient Problem* - the gradient decreases while reaching the input layer. So, as a result, the values near the input layer are almost not changing which is very bad for training. That's why a LSTM model should be used because it doesn't have that problem.

Այս խնդրի գաղափարը Deep Learning-ի միջոցով նոր անուններ գեներացնելն է։ Պարզ է, որ գործ ենք ունենալու տառերի հաջորդականությունների հետ, քանի որ անհանար է բառերի միջոցով նոր բառեր ստացվել։ Այսպես, կարող ենք հասկանալ, որ այս խնդրի **նշանների** (այսինքն տառերի) մակարդակի խնդիր է։ Պարզ ասած, քանի որ այս խնդրի հիմքը հաջորդականությունների օգտագործումն է, պարզ է դառնում, որ այստեղ անհրաժեշտ է օգտագործել **RNN** (Հերթական Նյարդային Ցանց)։ Նաև, քանի որ հնարավոր չի տրամադրել ցանցին ոչ մի $\bm{X}$ and $\bm{y}$ (այսւնքն համապատասխանաբար *data* և *target*), սա կլինկ գեներատոր, որը պիտի աշխատի տառ առ տառ, բայց նաև հաշվի առնելով նախորդ տպած տառերը։
Հիմնականում նոր անուն ստեծցելը դա այն խնդիրը չի, որտեղ Short Term Memory Problem-ը կարող է որոշակի դեր խաղա, որովհետև անունները սովորաբար այդքան երկար չեն լինում։ Ուստի, կարոց ենք օգտագործել հասարակ RNN։ Բայց իրականում մի ոիրիշ խնդրիր էլ է տեղի ունենում։ Ամենակարևոր բանը, ինչի պատճառով է առաջանում *Short Term Memory Problem*-ը, *Vanishing Gradient Problem*-ն է։ Նյարդային ցանցի ուսումը նշանակում ոչ այլ ինչ, ինչ իր պարամետրերի ճշտգրտումը։ Նյարդային ցանցը փորձուկ է կանխատեսում անել և նա համեմատում է իր կանխատեսումը  իրական պատասխանի հետ, որպեսզի հաջորդ անգամ փորձի ավելի լավ և դիպուկ կանխատեսում անի։ Իսկ գրադիենտը, որին հաշվարկվում է ցանցը, ցույց է տալիս, թե ցացը որքան  պիտի ճշտգրտի իր պարամետրերը հետ տարածման (back-propagation) ժամանակ։ Եվ ահա առաջանում է Vanishing Gradient Problem-ը՝ գրադիենտը հասնելով մուտքային շերտին նվազում է։ Եվ արդյունքոյմ մուտքային շերտի մոտ գտնվող ցուցմունքները գրեթե չեն փոխվում, ինչը շատ վատ է ցանցի ուսման համար։ Հետևաբար իրականացնելու համար ընտրված է LSTM մոդելը, որը չունի այդ պրոբլեմ։

## Preparing the data | Տվյալների պատրաստում
First of all, we should prepare a set of all characters which were used in the dataset. Sure, there is much better way to get that set but as we know that all characters of the Armenian alphabet were used we can define the following string:
Առաջին հերթին, պետք է պատրաստենք բոլոր նիշերի հավաքածուն, որոնք օգտագործվել են տվյալների բազայում: Իհարկե, այդ հավաքածուն ստանալու շատ ավելի լավ միջոցներ կան, բայց քանի որ գիտենք, որ օգտագործվել են հայերեն այբուբենի բոլոր նիշերը, կարող ենք սահմանել հետևյալ տողը.

> all_characters = "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքևօֆ\n"

It will be used for translating characters to numeric value.
Սա կօգտագործվի նիշերը թվեր թարգմանելու համար:

Note: as you can see all letters are lowercase and there is an additional symbol `\n`. All the names in the dataset were lowercased, it was not necessary though - it was just done for making the task case-independent. And the `\n` symbol is used because the sequence is just a set of characters, so the dataset is actually a long line of characters where the names are separated by `\n`, so it also should be in the dataset.
Նշում. ինչպես տեսնում եք, բոլոր տառերը փոքրատառ են և կա լրացուցիչ `\n` նշան: Տվյալների հավաքածուի բոլոր անունները սարքած են փոքրատառ, սակայն դա անհրաժեշտ չէր. դա արվել էր պարզապես խնդիրը մեծատառերից անկախ դարձնելու համար: Իսկ `\n` խորհրդանիշն օգտագործվում է, քանի որ հաջորդականությունը պարզապես նիշերի մի շարք է, ուստի տվյալների հավաքածուն իրականում նիշերի երկար տող է, որտեղ անուններն առանձնացված են `\n`-ով, դրա համար այն նույնպես պետք է լինի տվյալների հավաքածուում:

## Model | Մոդել
For implementing the LSTM model we should create two classes: ***RNN*** and ***Generator***. RNN is the actual model of the neural net, i.e. the sizes of input and output layers, number of layers and so on. For this task we can use a simple LSTM model which consists of 3 parts:
* Embedding layer (used for preparing the sequence of characters for the neural net);
* LSTM-layer (used for predictions);
* Fully-Connected layer (used for output).

LSTM մոդելը իրականացնելու համար պիտի ստեղծենք երկու կլաս՝ ***RNN*** և ***Generator***։ RNN-ը հենց նեյրոնային ցանցի մոդելն է, այսինքն իր մտքային և ելքային շերտերին չափերը, թաքնված շերտերի քանակը և այլն։  Այս խնդրի համար կարող ենք օգտագործել հասարակ LSTM մոդել, որը բաղկացած է 3 մասից՝
* Embedding layer (Օգտագործվում է տվյալները ցանցի պատրաստելու համար)
* LSTM-layer (Օգտագործվում է կանկատեսումների համար)
* Fully-Connected layer (Օգտագործվում է ելքի համար)

And the Generator will be responsible for translating the characters to numbers, selecting batches for the training and the training itself. Here are the parameters for the net:
Իսկ Generator-ը ապահովում են նիշերը թվերն թարգմանությունը, batch-եր ընտրելը և հենց ուսումը։ Ահա ցամցի պարամետրերը՝

    # այս պարամետրերը հենց այս տվյալների համար են, քանի որ անունների քանակը բավականին քիչ է
    self.chunk_len = 250
    self.num_epochs = 200
    self.batch_size = 1
    self.print_every = 50
    self.hidden_size = 256
    self.num_layers = 2
    self.lr = 0.003

So, basically every time the net selects some sequence from the dataset which is `chunk_len` long. Also we should train the net several times to get the best one, so `num_epochs` is how many times the net will try to adjust itself. The `batch_size` is how many samples the net sees before updating the model. So, it this case it updates its values at every epoch.
Այսպիսով, հիմնականում ամեն անգամ, երբ ցանցը ընտրում է որոշ հաջորդականություն տվյալների հավաքածուից, որի երկարությունը `chunk_len` է: Նաև մենք պետք է մի քանի անգամ ցանցը ուսուցել լավագույնը ստանալու համար, ուստի `num_epochs`-ն այն է, թե քանի անգամ ցանցը կփորձի փոխել ինքն իրեն: `batch_size`-ն այն է, թե քանի sample է ցանցը տեսնում մինչև մոդելը թարմացնելը: Այսպիսով, այս դեպքում այն թարմացնում է իր պարամետրերը ամեն epoch-ի ժամանակ:

## Training | Ուսուցում
This section shows how the model is trained.
Այս բաժինը ցույց է տալիս, թե ինչպես է մոդելը սովորում:

First of all, we create a RNN model, which was defined previously. Also, we need to create an optimizer (*Adam*) and criterion (*Cross-entropy Loss* or *Log Loss*) for the training.
Առաջին հերթին ստեղծում ենք RNN մոդելը, որը նախապես ներկայացված էր։ Բացի այդ, պետք է ստեղծենք օպտիմիզատոր (*Adam*) և չափանիշ (criterion) (*Cross-entropy Loss* կամ *Log Loss*) սովորելու համար։

	# ցանցի ուսումը
	def train(self):
		self.rnn = RNN(n_characters, self.hidden_size, self.num_layers, n_characters).to(device)
		optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr)
		criterion = nn.CrossEntropyLoss()
    
   And then for every epoch:
   Եվ հետո յուրաքանչյուր epoch-ի համար.

	for epoch in range(1, self.num_epochs + 1):
		inp, target = self.get_random_batch()
		hidden, cell = self.rnn.init_hidden(batch_size=self.batch_size)
    
		self.rnn.zero_grad()
		loss = 0
		inp = inp.to(device)
		target = target.to(device)
    
	    for c in range(self.chunk_len):
		    output, (hidden, cell) = self.rnn(inp[:, c], hidden, cell)
			loss += criterion(output, target[:, c])
		
		loss.backward()
		optimizer.step()
		
		loss = loss.item()/self.chunk_len
		if epoch % self.print_every == 0:
			print(f"Loss: {loss}")
			print(self.generate())

As you can see, for every time the net tries to adjust itself, it firstly gets a random batch (i.e. a random sequence from the given dataset) and separates it into *input* and *target* values. After, it initializes its *hidden* and *cell* values which are responsible for remembering the data. Then, it calculates the loss for every character comparing it with the *target* value and after that the back-propagation starts.
Ինչպես տեսնում եք, ամեն անգամ, երբ ցանցը փորձում է ճշտգրտել ինքն իրեն, այն առաջին հերթին ստանում է պատահական խմբաքանակ (batch) (այսինքն՝ պատահական հաջորդականություն տվյալ տվյալների բազայից) և այն բաժանում է *input* և *target*։ Այնուհետև այն սկզբնավորում է իր *hidden* և *cell* ցուցանքները, որոնք ապահովում են տվյալների հիշելը: Հետո այն հաշվարկում է կորուստը յուրաքանչյուր նիշի համար՝ համեմատելով այն *target*-ի հետ և դրանից հետո սկսվում է հետ-տարածումը (back-propagation)։

## Result | Արդյունք

    ահ
    վամիկ
    վիես
    վրան
    ագնասն
    վոսրես
    վարիրիրե
    վարլոր
    եղա
    անիս
    վասի
    վեմու
    ավես
    աթել
    նապիտո
    մեղիոս
    ալիոս
    դեստապամ
    ռաման
    վազավե
    ա
    վավեն
    աստիրանիրինե
    անդրամին
    ելենեա
    սան
    եվա
    ազատ
    մկթոմ
    հռիթեոսմ
    մղուկաս
    տիթան
    մռապել
    ամիսատ
    ապիթ
    ավանո
    արկամար
    բարակ
    արարիատ
    արսիս
    արպիստակ
    միլիտ
The net gives pretty good result. It's worth to note that the net took into account the peculiarities of the Armenian language. The neural net never knew that the `ւ` letter is written only after `ո` or `ե` but all the results are good and the net is **character**-based.
Ցանցը բավականին լավ արդյունք է տալիս։ Հարկ է նշել, որ ցանցը հաշվի է առել հայերենի առանձնահատկությունները։ Նեյրոնային ցանցը երբեք չի իմացել, որ `ւ` տառը գրվում է միայն `ո`-ից կամ `ե`-ից հետո, սակայն բոլոր արդյունքները բավականին լավն են, և ցանցը հիմնված է **նիշերի** վրա: