# train_model.py - emotion classifier for journal entries
# this will be used to identify what emotion the user is feeling
# in their journal entry and then use it to return the 
# correct reframe and affirmation to the emotion

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

texts = [

    # overwhelmed (20)
    "i feel overwhelmed with college work and deadlines",
    "there is too much to do and i don’t know where to start",
    "everything feels heavy and stressful and i keep delaying things",
    "i feel pressure and it's making me shut down",
    "my mind feels overloaded and i can’t think clearly",
    "i feel like everything is piling up faster than i can handle",
    "i feel like i’m drowning in responsibilities",
    "my brain feels cluttered and tense",
    "i feel like i can’t keep up with everything expected from me",
    "i feel overloaded and mentally flooded",
        "i feel completely buried under all my tasks",
    "i open my laptop and instantly feel like it is too much",
    "i look at my to do list and want to cry from overwhelm",
    "even small tasks feel huge because everything has piled up",
    "i feel like i am spinning and not getting anywhere",
    "my brain feels like it is buffering with too many tabs open",
    "i keep jumping between tasks because i feel so overloaded",
    "i feel like i am drowning in college work and responsibilities",
    "there are so many deadlines that my body just wants to shut down",
    "i feel like i could scream because everything is demanding my attention",

    # anxious (20)
    "i am worried about failing and it makes me freeze",
    "i feel nervous and tense about starting this task",
    "i keep imagining the worst and it stops me from working",
    "i feel panicky when i think about assignments",
    "i am afraid of making mistakes so i avoid starting",
    "i feel uneasy and unsure about how things will turn out",
    "i am scared that i won’t do well enough",
    "i feel jumpy and on edge about getting things done",
    "i feel like something will go wrong if i start",
    "i feel tense and unsettled about my work",
    "i feel sick with nerves about starting this assignment",
    "my heart races when i think about opening my notes",
    "i keep imagining myself failing and it makes me panic",
    "i feel on edge and jittery when i sit down to work",
    "my chest feels tight and my hands feel shaky about this task",
    "i am scared that whatever i do will not be good enough",
    "i feel worried that i have already left it too late",
    "i overthink every little step and it makes me afraid to move",
    "i feel anxious that people will judge my work harshly",
    "i feel like something bad will happen if i even try to start",

    # stuck (20)
    "i feel stuck and can’t get myself to begin",
    "i sit here knowing what to do but not doing it",
    "i feel blocked and unable to take action",
    "i keep putting things off even though i want to start",
    "i feel frozen and unable to move forward",
    "i feel trapped in hesitation",
    "i feel like i can’t get myself to start even a tiny bit",
    "i feel stuck staring at the work but not doing it",
    "i feel paused and unable to take the first step",
    "i feel like i'm glued in place and not moving",
    "i am staring at the screen and nothing is coming out",
    "i keep rereading the same line and not doing anything",
    "i feel frozen and unable to take the first step",
    "i keep scrolling on my phone instead of starting",
    "i know what i should do but i am not moving",
    "i feel like i am in quicksand and every action feels impossible",
    "i keep opening and closing the same document without progress",
    "i feel mentally blocked and can’t get into the task",
    "i am stuck in planning and never actually doing the thing",
    "i feel stuck in a loop of thinking about it but not acting",

    # stressed (20)
    "i feel stressed because i am carrying so much",
    "i feel tense and stretched too thin",
    "i feel like everything is urgent and critical",
    "i feel pressured and my body feels tight",
    "i feel like time is running out and it stresses me",
    "i feel like i can't relax because there's too much to do",
    "i feel wound up and overloaded with tasks",
    "i feel stressed trying to balance everything",
    "i feel like my head is buzzing from responsibility",
    "i feel tense and under pressure to perform",
    "my body feels tight and tense from all the stress",
    "i feel like my shoulders are up by my ears from worrying",
    "i keep clenching my jaw because I am so stressed",
    "my mind keeps racing through everything i have to do",
    "i feel under pressure and my patience is really low",
    "i feel snappy and irritable because of how stressed i am",
    "i can’t relax because my brain keeps reminding me of deadlines",
    "i feel like i am constantly rushing and never catching up",
    "i feel drained from constantly thinking about all my tasks",
    "i feel stress buzzing in the background all day long",

    # tired (20)
    "i feel tired and mentally drained",
    "i feel exhausted from thinking so much",
    "i feel like my energy is gone even before i start",
    "i feel worn out and unable to focus",
    "i feel like i need rest before i can continue",
    "i feel sleepy and unfocused when i try to work",
    "i feel like my brain is running on empty",
    "i feel wiped out after a long day",
    "i feel fatigue slowing me down",
    "i feel like i don’t have the energy to try",
    "i feel exhausted even after sleeping",
    "my eyes feel heavy and i just want to lie down",
    "i feel drained and like i have no fuel left",
    "my body feels slow and heavy when i try to work",
    "i just want to nap instead of doing anything",
    "i feel like yawning every five seconds when i open my laptop",
    "i feel worn out and mentally foggy all day",
    "i feel like i have zero energy to give to this task",
    "i feel like my brain is moving through thick fog",
    "i could fall asleep at my desk because i am so tired",

    # calm (20)
    "i feel calm and grounded and ready to take things slowly",
    "my mind feels clear and relaxed right now",
    "i feel steady and comfortable working bit by bit",
    "i am relaxed and things feel manageable",
    "i feel peaceful and balanced about what i need to do",
    "i feel settled and okay with moving gently",
    "i feel at ease and centered while i work",
    "i feel calm and unhurried as i get things done",
    "i feel balanced and steady in my focus",
    "i feel relaxed and present while working",
    "i feel calm and steady enough to take one small step",
    "my breathing feels slow and relaxed as i work",
    "i feel grounded and safe even though i have things to do",
    "i feel quietly confident that i can handle this bit by bit",
    "i feel peaceful and unhurried about my tasks today",
    "i feel gentle towards myself while i make progress",
    "i feel composed and in control as i plan my work",
    "i feel light and at ease sitting at my desk",
    "i feel relaxed but still focused on what matters",
    "i feel okay taking it one tiny action at a time",
]

labels = (
    ["overwhelmed"] * 20 +
    ["anxious"] * 20 +
    ["stuck"] * 20 +
    ["stressed"] * 20 +
    ["tired"] * 20 +
    ["calm"] * 20
)


# convert text to tf-idf features
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # use single words + 2-word phrases
    min_df=1,             # keep all terms (small dataset)
    lowercase=True        # ensure everything is lowercased
)
X = vectorizer.fit_transform(texts)

# train logistic regression classifier
clf = LogisticRegression(max_iter=200)
clf.fit(X, labels)

# save model + vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(clf, "models/emotion_model.pkl")

print("Improved model trained and saved.")
print("Current working directory:", os.getcwd())
