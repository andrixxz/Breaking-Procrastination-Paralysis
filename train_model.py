# train_model.py - emotion classifier for journal entries
# expanded from 6 to 11 emotion classes for Phase 2 ML upgrade
# classes: overwhelmed, anxious, stuck, stressed, tired, calm, frustrated, guilty, unmotivated, hopeful and proud
# each class has 25 training samples (275 total)

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import os

texts = [

    # ===== OVERWHELMED (25) =====
    # volume overload, too many things, shutting down from quantity
    "i feel overwhelmed with college work and deadlines",
    "there is too much to do and i don't know where to start",
    "everything feels heavy and stressful and i keep delaying things",
    "i feel pressure and it's making me shut down",
    "my mind feels overloaded and i can't think clearly",
    "i feel like everything is piling up faster than i can handle",
    "i feel like i'm drowning in responsibilities",
    "my brain feels cluttered and tense",
    "i feel like i can't keep up with everything expected from me",
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
    "every time i finish one thing three more appear and i feel buried",
    "i have so many assignments due and i cannot process any of them",
    "the list of things i need to do is so long it makes me want to hide",
    "i feel like i am being crushed under the weight of everything at once",
    "my head is so full of tasks that i cannot focus on a single one",

    # ===== ANXIOUS (25) =====
    # fear, worry, nervousness about outcomes or starting
    "i am worried about failing and it makes me freeze",
    "i feel nervous and tense about starting this task",
    "i keep imagining the worst and it stops me from working",
    "i feel panicky when i think about assignments",
    "i am afraid of making mistakes so i avoid starting",
    "i feel uneasy and unsure about how things will turn out",
    "i am scared that i won't do well enough",
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
    "i keep worrying about what will happen if i get this wrong",
    "the fear of messing up is making it impossible to begin",
    "i feel a knot in my stomach every time i think about this project",
    "i am so nervous about submitting this that i keep putting it off",
    "my mind keeps playing worst case scenarios on repeat",

    # ===== STUCK (25) =====
    # frozen, unable to act, knowing what to do but not doing it
    "i feel stuck and can't get myself to begin",
    "i sit here knowing what to do but not doing it",
    "i feel blocked and unable to take action",
    "i keep putting things off even though i want to start",
    "i feel frozen and unable to move forward",
    "i feel trapped in hesitation",
    "i feel like i can't get myself to start even a tiny bit",
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
    "i feel mentally blocked and can't get into the task",
    "i am stuck in planning and never actually doing the thing",
    "i feel stuck in a loop of thinking about it but not acting",
    "i have been sitting here for an hour and have not typed a single word",
    "i want to start but something invisible is holding me back",
    "i feel paralysed and unable to do the simplest thing",
    "my body is at my desk but my brain refuses to engage with the work",
    "i keep telling myself i will start in five minutes but i never do",

    # ===== STRESSED (25) =====
    # pressure, tension, time urgency, body feels tight
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
    "i keep clenching my jaw because i am so stressed",
    "my mind keeps racing through everything i have to do",
    "i feel under pressure and my patience is really low",
    "i feel snappy and irritable because of how stressed i am",
    "i can't relax because my brain keeps reminding me of deadlines",
    "i feel like i am constantly rushing and never catching up",
    "i feel drained from constantly thinking about all my tasks",
    "i feel stress buzzing in the background all day long",
    "the deadline is tomorrow and i still have so much to do",
    "i feel like a rubber band about to snap from all the tension",
    "i can feel the stress in my neck and shoulders while i work",
    "there is not enough time and everything feels urgent right now",
    "i feel on edge because i have three things due this week",

    # ===== TIRED (25) =====
    # exhaustion, low energy, fatigue, wanting rest
    "i feel tired and mentally drained",
    "i feel exhausted from thinking so much",
    "i feel like my energy is gone even before i start",
    "i feel worn out and unable to focus",
    "i feel like i need rest before i can continue",
    "i feel sleepy and unfocused when i try to work",
    "i feel like my brain is running on empty",
    "i feel wiped out after a long day",
    "i feel fatigue slowing me down",
    "i feel like i don't have the energy to try",
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
    "i barely slept last night and now i cannot concentrate on anything",
    "my whole body feels heavy and sluggish today",
    "i have no mental energy left after this week",
    "i feel burnt out and like i need a week off to recover",
    "everything takes twice as long because i am so exhausted",

    # ===== CALM (25) =====
    # grounded, peaceful, steady, manageable
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
    "things feel manageable right now and i am at peace with my pace",
    "i feel a quiet stillness that makes it easy to focus",
    "i am in a good headspace and ready to work gently",
    "i feel centred and present and that is enough for now",
    "i feel safe and unhurried as i sit down to do some work",

    # ===== FRUSTRATED (25) =====
    # anger, irritation at self or situation, things not working
    "i am so frustrated with myself for not getting this done",
    "i keep trying and nothing is working and it makes me angry",
    "i feel annoyed that i wasted the whole day doing nothing",
    "i am irritated because i have been working on this and it still is not right",
    "i feel angry at myself for procrastinating again",
    "this is so frustrating because i know i can do better than this",
    "i feel like throwing my laptop because nothing is going right",
    "i am fed up with myself for leaving things to the last minute",
    "i feel frustrated that no matter what i try i keep getting stuck",
    "i keep making the same mistakes and it is driving me mad",
    "i am annoyed because i had a plan and it completely fell apart",
    "i feel irritated that this task is taking so much longer than expected",
    "i worked on this for hours and it is still not right and i want to scream",
    "i feel angry because everyone else seems to manage and i cannot",
    "i am frustrated because i keep going in circles with this assignment",
    "i feel like punching a wall because i just cannot figure this out",
    "i am so fed up with procrastinating that i am angry at myself",
    "nothing is clicking and i feel so annoyed about wasting my time",
    "i am furious with myself for letting things get this bad",
    "i tried my best and it still was not enough and that makes me livid",
    "i feel agitated because every time i fix one thing another breaks",
    "i keep hitting dead ends and my patience is completely gone",
    "i feel resentful that i have to do this when it feels pointless",
    "i am so annoyed that i cannot just sit down and get this done like a normal person",
    "the fact that i keep failing at this is making me more and more frustrated",

    # ===== GUILTY (25) =====
    # shame about avoidance, letting people down, wasting time
    "i feel guilty for not starting earlier when i had the chance",
    "i feel ashamed that i wasted the whole weekend doing nothing productive",
    "i feel bad because everyone else is working hard and i am doing nothing",
    "i let my group down by not finishing my part and i feel terrible",
    "i feel guilty that my parents are paying for college and i am wasting it",
    "i should have started this weeks ago and now i feel awful about it",
    "i feel like a bad person for putting this off for so long",
    "i feel ashamed of how behind i am compared to everyone else",
    "i promised myself i would work today and i didn't and now i feel guilty",
    "i feel terrible because my lecturer is counting on me and i have nothing done",
    "i keep letting people down and the guilt is eating me up inside",
    "i feel so ashamed that i spent the day on my phone instead of studying",
    "i should be further along by now and i feel awful about where i am",
    "i feel guilty every time i relax because i know i should be working",
    "i feel like i am disappointing everyone who believes in me",
    "i told my friend i would help them but i cannot even help myself and that feels terrible",
    "the shame of not doing anything today is sitting in my chest",
    "i feel bad because i know i am capable but i am not showing it",
    "i wasted another day and the guilt is overwhelming",
    "i feel ashamed because i keep saying i will change and i never do",
    "i feel terrible knowing i could have done more but chose not to",
    "i feel guilty for watching videos when i should have been studying",
    "i feel like i am letting my future self down by not working today",
    "i cancelled plans to work on my assignment but then did nothing and feel awful",
    "i feel so much shame about how little i have accomplished this week",

    # ===== UNMOTIVATED (25) =====
    # apathy, lack of drive, what is the point, no desire to engage
    "i just do not care about this assignment anymore",
    "i have zero motivation to do anything today",
    "i cannot find any reason to start working on this",
    "nothing feels worth the effort right now",
    "i feel completely disengaged from my work",
    "i used to care about this but now i feel nothing",
    "i cannot bring myself to care about this deadline",
    "everything feels pointless and i do not see why i should bother",
    "i feel flat and uninterested in anything academic",
    "i have no drive to open my laptop or look at my notes",
    "what is the point of doing this when it does not matter to me",
    "i feel empty when i think about my coursework",
    "i cannot find the spark to get started on anything",
    "my motivation is completely gone and i do not know how to get it back",
    "i feel disconnected from why i am even doing this degree",
    "i look at my work and feel absolutely nothing about it",
    "i do not want to do this and i do not care if it gets done",
    "i feel numb towards everything i need to do",
    "even things i used to enjoy feel like a chore now",
    "i have lost all interest in this project and i cannot pretend otherwise",
    "i feel like i am going through the motions with no real purpose",
    "nothing excites me about my work anymore and that is scary",
    "i wake up with no intention of doing anything productive",
    "i feel like a robot just existing without any drive or passion",
    "i cannot remember the last time i felt motivated to do my work",

    # ===== HOPEFUL (25) =====
    # cautious optimism, seeing possibility, things might work out
    "i feel like maybe i can get through this after all",
    "something shifted today and i feel a bit more positive about my work",
    "i think things might actually work out if i keep going",
    "i feel a small spark of hope that i can turn this around",
    "i am starting to believe that i can handle this one step at a time",
    "today felt different and i feel like i am on the right path",
    "i noticed some progress and it makes me think i can do this",
    "i feel cautiously optimistic about my ability to catch up",
    "for the first time in a while i feel like things are possible",
    "i feel like the fog is lifting and i can see a way forward",
    "i woke up today feeling like i might actually be able to do this",
    "i feel encouraged because i managed to start something small yesterday",
    "i am beginning to see that small steps really do add up",
    "i feel a gentle sense of possibility that was not there before",
    "something about today makes me feel like i am going to be okay",
    "i feel lighter today and more open to trying",
    "i think i can do this if i just take it one thing at a time",
    "i feel like i am slowly getting better at handling things",
    "there is a small part of me that believes i can still make this work",
    "i feel a flicker of confidence that was not there yesterday",
    "i feel like maybe i have been too hard on myself and i can actually do this",
    "i am starting to trust that showing up counts even when progress is slow",
    "i feel hopeful that this time will be different from the other times",
    "i can see a path forward now and that gives me a bit of courage",
    "i feel a quiet sense of possibility growing inside me today",

    # ===== PROUD (25) =====
    # accomplishment, satisfaction, recognising own effort
    "i actually did it and i feel proud of myself",
    "i finished something today and it feels really good",
    "i showed up and did the work even though i did not want to",
    "i feel proud because i pushed through the resistance and started",
    "i completed my assignment and i am genuinely pleased with myself",
    "i stuck with it even when it was hard and that means something",
    "i feel a sense of accomplishment that i have not felt in a while",
    "i did more today than i expected and i feel great about it",
    "i am proud of myself for not giving up when it got difficult",
    "i managed to focus for a full hour and that is a win",
    "i feel satisfied with the progress i made today even though it was small",
    "i finally submitted the thing i have been avoiding and i feel so relieved and proud",
    "i kept my promise to myself and worked on my project today",
    "i feel accomplished because i broke the cycle of avoidance",
    "i did something i was dreading and it actually went well",
    "i am proud that i asked for help instead of suffering alone",
    "i showed up for myself today and that is enough to feel good about",
    "i wrote more than i thought i could and i feel really proud",
    "i managed to start even though every part of me wanted to avoid it",
    "i feel a warm sense of pride because i chose to try today",
    "i did not do it perfectly but i did it and i am proud of that",
    "today i proved to myself that i am capable of doing hard things",
    "i am proud because i stayed consistent this week even when it was tough",
    "i feel good about myself because i took a step forward instead of standing still",
    "i handled today better than i thought i would and that makes me proud",
]

labels = (
    ["overwhelmed"] * 25 +
    ["anxious"] * 25 +
    ["stuck"] * 25 +
    ["stressed"] * 25 +
    ["tired"] * 25 +
    ["calm"] * 25 +
    ["frustrated"] * 25 +
    ["guilty"] * 25 +
    ["unmotivated"] * 25 +
    ["hopeful"] * 25 +
    ["proud"] * 25
)

# sanity check: texts and labels must match
assert len(texts) == len(labels), f"Mismatch: {len(texts)} texts vs {len(labels)} labels"
print(f"Training samples: {len(texts)} ({len(set(labels))} classes, {len(texts)//len(set(labels))} per class)")

# convert text to tf-idf features
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),   # use single words + 2-word phrases
    min_df=1,             # keep all terms (small dataset)
    lowercase=True        # ensure everything is lowercased
)
X = vectorizer.fit_transform(texts)

# train logistic regression classifier
clf = LogisticRegression(max_iter=500, C=1.0)
clf.fit(X, labels)

# cross-validation to verify model quality
scores = cross_val_score(clf, X, labels, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})")

# save model + vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(clf, "models/emotion_model.pkl")

print("Expanded model trained and saved (11 emotion classes).")
print(f"Classes: {sorted(set(labels))}")


# =====================================================================
# MODEL 2: BEHAVIOUR STATE CLASSIFIER
# classifies WHAT the user is doing, not how they feel
# 6 classes: avoidance, overwhelm, action, completion, recovery, rumination
# each class has 25 training samples (150 total)
# separate TF-IDF vectorizer + Logistic Regression pipeline
# =====================================================================

print("\n--- Training Behaviour State Classifier (Model 2) ---")

behaviour_texts = [

    # ===== AVOIDANCE (25) =====
    # actively dodging, distracting, postponing, choosing not to engage
    "i have been scrolling on my phone for two hours instead of working",
    "i keep finding other things to do so i do not have to start the assignment",
    "i told myself i would start after lunch but lunch was three hours ago",
    "i cleaned my entire room just to avoid opening my laptop",
    "i keep putting it off and doing literally anything else instead",
    "i watched four episodes of a show i do not even like to avoid studying",
    "i keep saying i will do it tomorrow but tomorrow never comes",
    "i have been reorganising my desktop files instead of writing my essay",
    "i made three cups of tea today just to delay sitting down to work",
    "i went for a walk to clear my head but really i was just running away from the task",
    "i keep checking social media because i do not want to face my coursework",
    "i decided to rearrange my bookshelf instead of studying for the exam",
    "i keep starting different tasks so i never actually finish the important one",
    "i told my friend i was busy but really i was just avoiding the work",
    "i spent the whole morning online shopping instead of doing my project",
    "i have been playing games on my phone because i do not want to think about my essay",
    "i keep pretending i have other priorities so i do not have to start",
    "i am deliberately choosing easier tasks to avoid the hard one",
    "i signed up for a webinar i do not need just to have an excuse not to work",
    "i keep opening the document and then immediately closing it again",
    "i told myself i need more research before i can start but that is just an excuse",
    "i have been avoiding my emails because one of them is about the deadline",
    "i am distracting myself with cooking and cleaning because the task feels too big",
    "i am finding every possible reason not to start this right now",
    "i left the library early because i did not want to face the assignment",

    # ===== OVERWHELM (25) =====
    # paralysed by volume or complexity, frozen, unable to choose where to begin
    "there are so many things to do that i cannot even pick one to start with",
    "i opened my to do list and immediately felt paralysed by how long it is",
    "i do not know where to begin because everything feels equally urgent",
    "the assignment brief is so long and complex that i froze just reading it",
    "i have five deadlines this week and i cannot process any of them",
    "my brain shut down when i looked at everything i need to do",
    "i tried to plan my week but the volume of work made me go blank",
    "every time i think about starting i get hit with how much there is to do",
    "i sat down to work but the sheer amount of it made me freeze up",
    "i feel paralysed because i cannot figure out what to prioritise first",
    "the project has so many parts that i do not know which one to tackle",
    "i opened three tabs for different assignments and then just stared at them",
    "the complexity of this task is making my brain shut off completely",
    "i have so many competing deadlines that i ended up doing none of them",
    "i tried to break the task down but even the pieces felt too big",
    "i looked at the exam timetable and my mind went completely blank",
    "i cannot even write a to do list because there is too much to capture",
    "the number of things i need to do today is making me freeze",
    "i feel buried under all the requirements and cannot move",
    "my brain is in shutdown mode because it cannot handle the workload",
    "i have been staring at the blank page because the whole thing feels too big to start",
    "the scope of this project is so large that i do not know where to even begin",
    "i keep trying to figure out a plan but there is just too much to organise",
    "everything hit me at once and now i cannot do any of it",
    "i feel frozen looking at everything that needs to be done this month",

    # ===== ACTION (25) =====
    # actively working, engaging, making progress, doing the thing
    "i just sat down and started writing and it is going okay",
    "i am working on my assignment right now and making progress",
    "i opened the document and started typing even though i did not feel ready",
    "i set a timer for twenty minutes and i am actually doing the work",
    "i started the first paragraph and it is not perfect but at least i started",
    "i am reading through the material and taking notes as i go",
    "i broke the task into small steps and i am working through them now",
    "i am actively working on this even though part of me wants to stop",
    "i managed to start the introduction and i am going to keep writing",
    "i am making progress on my project one small piece at a time",
    "i just finished outlining my essay and now i am filling in the sections",
    "i am coding the first feature of my project and it feels good to be doing something",
    "i opened my notes and started reviewing for the exam",
    "i am writing even though it is not great because getting something down matters",
    "i decided to just start with the easiest part and build from there",
    "i have been working for thirty minutes straight which is more than yesterday",
    "i am in the middle of drafting my report and making steady progress",
    "i just submitted a draft to my lecturer for feedback",
    "i picked one task from my list and i am focusing on just that one thing",
    "i started researching for my project and have found some good sources",
    "i am halfway through the assignment and taking a short break before continuing",
    "i am actively engaging with the material even though it is hard",
    "i forced myself to write the first sentence and now the rest is flowing",
    "i am working through the problem set one question at a time",
    "i am doing the thing i have been putting off and it is not as bad as i thought",

    # ===== COMPLETION (25) =====
    # finished something, reflecting on accomplishment, task done
    "i finished my essay and submitted it before the deadline",
    "i completed all my tasks for today and i can finally relax",
    "i just handed in the assignment that i have been dreading for weeks",
    "i got through everything on my to do list today",
    "i finished the project and it actually turned out better than i expected",
    "i submitted my report and it feels like a weight has been lifted",
    "i completed the coding challenge and all the tests passed",
    "i managed to finish the entire chapter review before the end of the day",
    "i ticked off the last item on my list and i am done for today",
    "i just sent the final version to my group and our project is complete",
    "i finished studying for the exam and covered all the topics",
    "i completed the presentation slides and they look good",
    "i handed in my coursework and i do not have to think about it anymore",
    "i got through all three tasks i set for myself today",
    "i wrapped up the lab report and emailed it to my lecturer",
    "i finished writing all the unit tests for my project",
    "i completed the literature review section of my dissertation",
    "i am done with my work for the day and everything is submitted",
    "i finalised the design and sent it for review",
    "i finished reading all the papers i needed for my research",
    "i completed the assignment ahead of the deadline for once",
    "i finally finished that task that has been hanging over me all week",
    "i just crossed the last thing off my list and it feels amazing",
    "i managed to complete the full set of exercises before class tomorrow",
    "i submitted everything that was due and i have nothing left outstanding",

    # ===== RECOVERY (25) =====
    # resting, recharging, self-care, taking a deliberate break after effort
    "i am taking a break because i worked hard this morning and i need to recharge",
    "i went for a walk after finishing my work to clear my head",
    "i am resting now because i know i will work better if i recover first",
    "i am giving myself permission to do nothing for a while after a long day",
    "i took a nap after studying and i feel much better now",
    "i am watching something relaxing because i earned a break today",
    "i am sitting quietly with a cup of tea after a productive morning",
    "i decided to stop working for the day because i have done enough",
    "i am letting myself rest without guilt because rest is part of the process",
    "i stepped away from my desk to take care of myself for a bit",
    "i am taking the evening off because i was focused all afternoon",
    "i cooked a proper meal instead of skipping it like i usually do when busy",
    "i am journaling to wind down after a long study session",
    "i went outside for fresh air because i have been at my desk all day",
    "i am listening to music and doing nothing and that is okay right now",
    "i took a long shower to relax after finishing my assignment",
    "i am deliberately resting now so i have energy for tomorrow",
    "i put my phone away and just sat in silence for ten minutes to reset",
    "i am reading something for fun because i need a mental break from coursework",
    "i told myself it is okay to stop for today and pick it up fresh tomorrow",
    "i am stretching and moving my body after sitting for hours studying",
    "i finished what i could and now i am winding down for the night",
    "i treated myself to something nice because i worked really hard today",
    "i am taking the afternoon off because pushing through will just burn me out",
    "i am resting and that is a productive choice right now",

    # ===== RUMINATION (25) =====
    # overthinking, stuck in mental loops, analysing without acting
    "i keep thinking about the assignment but i am not actually doing anything about it",
    "i have been going over the same thoughts in my head for hours",
    "i cannot stop thinking about how badly this might go",
    "i keep replaying what i should have done differently and it is consuming me",
    "my mind is going in circles about whether i am good enough for this",
    "i have been overthinking every single part of this task instead of starting",
    "i keep analysing all the ways this could go wrong instead of just trying",
    "i am stuck in my head thinking about the deadline but not doing anything",
    "i have been mentally debating which task to start for over an hour",
    "i keep worrying about the outcome instead of focusing on the process",
    "my thoughts are spiralling and i cannot break out of the loop",
    "i keep going back and forth in my mind about whether to start now or later",
    "i have been obsessing over the details instead of just getting something done",
    "i cannot stop comparing myself to others and it is keeping me from working",
    "i have been sitting here thinking about everything that could go wrong",
    "my brain is stuck replaying the same worries over and over again",
    "i keep second guessing every decision i make about this project",
    "i have been mentally planning for so long that i never actually start doing",
    "i am caught in a loop of thinking about how much time i have already wasted",
    "i keep going over my mistakes in my head instead of moving forward",
    "i am overthinking whether my approach is right instead of just trying it",
    "my mind keeps cycling through worst case scenarios on repeat",
    "i cannot stop mentally calculating how behind i am compared to others",
    "i have been debating with myself about the best way to do this for ages",
    "i keep thinking i need a perfect plan before i can start and it is keeping me stuck",
]

behaviour_labels = (
    ["avoidance"] * 25 +
    ["overwhelm"] * 25 +
    ["action"] * 25 +
    ["completion"] * 25 +
    ["recovery"] * 25 +
    ["rumination"] * 25
)

# sanity check
assert len(behaviour_texts) == len(behaviour_labels), \
    f"Mismatch: {len(behaviour_texts)} texts vs {len(behaviour_labels)} labels"
print(f"Training samples: {len(behaviour_texts)} ({len(set(behaviour_labels))} classes, "
      f"{len(behaviour_texts)//len(set(behaviour_labels))} per class)")

# separate TF-IDF vectorizer for behaviour model (different feature space)
behaviour_vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    lowercase=True
)
X_behaviour = behaviour_vectorizer.fit_transform(behaviour_texts)

# train logistic regression for behaviour states
behaviour_clf = LogisticRegression(max_iter=500, C=1.0)
behaviour_clf.fit(X_behaviour, behaviour_labels)

# cross-validation
behaviour_scores = cross_val_score(behaviour_clf, X_behaviour, behaviour_labels, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {np.mean(behaviour_scores):.2f} (+/- {np.std(behaviour_scores):.2f})")

# save behaviour model + vectorizer
joblib.dump(behaviour_vectorizer, "models/behaviour_vectorizer.pkl")
joblib.dump(behaviour_clf, "models/behaviour_model.pkl")

print("Behaviour state classifier trained and saved (6 classes).")
print(f"Classes: {sorted(set(behaviour_labels))}")
print(f"TF-IDF features: {X_behaviour.shape[1]}")
