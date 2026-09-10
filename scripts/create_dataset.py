import os
import pandas as pd

DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)
FILE_PATH = os.path.join(DATA_DIR, "philosophy_books.csv")

books_data = [
    # === 1. EXISTENTIALISM & ABSURDISM ===
    {
        "id": 1,
        "title": "The Myth of Sisyphus",
        "author": "Albert Camus",
        "school": "Absurdism",
        "summary": "Addresses the fundamental philosophical question of suicide and finding meaning in an indifferent, godless universe through defiance, rebellion, and personal freedom.",
        "key_concepts": "absurd, suicide, rebellion, scorn of the gods, defying fate, freedom"
    },
    {
        "id": 2,
        "title": "The Stranger",
        "author": "Albert Camus",
        "school": "Absurdism",
        "summary": "Follows Meursault, an emotionally detached man who refuses to conform to societal rituals and moral expectations, embracing emotional truth until his execution.",
        "key_concepts": "detachment, honesty, social conformity, absurdity, authentic life"
    },
    {
        "id": 3,
        "title": "Being and Nothingness",
        "author": "Jean-Paul Sartre",
        "school": "Existentialism",
        "summary": "Argues that human existence precedes essence. Humans are condemned to radical freedom and must take complete responsibility for defining their own identity without divine guidance.",
        "key_concepts": "bad faith, radical freedom, existence precedes essence, anguish, self-definition"
    },
    {
        "id": 4,
        "title": "Nausea",
        "author": "Jean-Paul Sartre",
        "school": "Existentialism",
        "summary": "A philosophical novel exploring an historian's visceral dread upon realizing that the external world and existence itself are entirely contingent and devoid of intrinsic meaning.",
        "key_concepts": "existential dread, contingency, meaninglessness, alienation, physical revulsion"
    },
    {
        "id": 5,
        "title": "Fear and Trembling",
        "author": "Søren Kierkegaard",
        "school": "Christian Existentialism",
        "summary": "Examines Abraham's sacrifice of Isaac, exploring the terrifying leap of faith, the suspension of the ethical, and the solitary individual confronting divine commands.",
        "key_concepts": "leap of faith, anxiety, teleological suspension of the ethical, knight of faith"
    },
    {
        "id": 6,
        "title": "The Sickness Unto Death",
        "author": "Søren Kierkegaard",
        "school": "Christian Existentialism",
        "summary": "Investigates human despair as a spiritual failure of the self to relate authentically to itself and to the transcendent power that created it.",
        "key_concepts": "despair, selfhood, alienation, spiritual sickness, authentic self"
    },
    {
        "id": 7,
        "title": "The Ethics of Ambiguity",
        "author": "Simone de Beauvoir",
        "school": "Existentialism",
        "summary": "Develops a moral theory grounded in existentialist freedom, balancing personal autonomy with the ethical responsibility to liberate others from oppression.",
        "key_concepts": "freedom of others, ethics, ambiguity of existence, anti-oppression, moral action"
    },

    # === 2. INDIVIDUALISM & WILL / OVERCOMING ===
    {
        "id": 8,
        "title": "Thus Spoke Zarathustra",
        "author": "Friedrich Nietzsche",
        "school": "Nietzscheanism",
        "summary": "A prophetic narrative proclaiming the death of God, overcoming traditional morality, embracing eternal recurrence, and striving towards the self-transcending Übermensch.",
        "key_concepts": "death of god, ubermensch, eternal recurrence, defying gods, will to power, self-overcoming"
    },
    {
        "id": 9,
        "title": "Beyond Good and Evil",
        "author": "Friedrich Nietzsche",
        "school": "Nietzscheanism",
        "summary": "Deconstructs traditional moral philosophy, criticizing dogmatic truths and introducing the distinction between master and slave moralities.",
        "key_concepts": "master-slave morality, will to power, perspectivism, critique of dogma, autonomous values"
    },
    {
        "id": 10,
        "title": "On the Genealogy of Morals",
        "author": "Friedrich Nietzsche",
        "school": "Nietzscheanism",
        "summary": "Traces the psychological origins of guilt, bad conscience, and ascetic ideals, exposing how religious resentment shaped modern moral systems.",
        "key_concepts": "ressentiment, bad conscience, guilt, ascetic ideals, critique of christianity"
    },
    {
        "id": 11,
        "title": "The Gay Science",
        "author": "Friedrich Nietzsche",
        "school": "Nietzscheanism",
        "summary": "Celebrates intellectual freedom, courage, and joyful experimentation in thinking, featuring the famous declaration that God is dead.",
        "key_concepts": "amor fati, joyful wisdom, intellectual honesty, death of god, love of fate"
    },
    {
        "id": 12,
        "title": "The World as Will and Representation",
        "author": "Arthur Schopenhauer",
        "school": "Philosophical Pessimism",
        "summary": "Posits that the ultimate reality is a blind, insatiable Will that produces endless suffering, manageable only through ascetic contemplation and artistic detachment.",
        "key_concepts": "metaphysical will, pessimism, suffering, aesthetic contemplation, cessation of desire"
    },
    {
        "id": 13,
        "title": "The Ego and Its Own",
        "author": "Max Stirner",
        "school": "Egoist Anarchism",
        "summary": "Rejects all religious, ideological, and moral concepts as mental illusions or spooks, advocating radical self-ownership and absolute individual interest.",
        "key_concepts": "spooks of the mind, radical egoism, self-ownership, anti-statism, individual supremacy"
    },

    # === 3. STOICISM & INNER AUTONOMY ===
    {
        "id": 14,
        "title": "Meditations",
        "author": "Marcus Aurelius",
        "school": "Stoicism",
        "summary": "Personal journal of a Roman Emperor meditating on duty, accepting mortality, emotional self-mastery, and maintaining an unshakeable inner citadel.",
        "key_concepts": "inner citadel, duty, impermanence, controlling reactions, rational cosmos, emotional calm"
    },
    {
        "id": 15,
        "title": "Letters from a Stoic",
        "author": "Seneca",
        "school": "Stoicism",
        "summary": "Practical letters offering timeless guidance on handling grief, avoiding the trap of materialism, using time deliberately, and conquering the fear of death.",
        "key_concepts": "fear of death, value of time, tranquility, mental resilience, overcoming luxury"
    },
    {
        "id": 16,
        "title": "Discourses and Selected Writings",
        "author": "Epictetus",
        "school": "Stoicism",
        "summary": "A former slave teaches that true freedom resides in recognizing what is within our control and abandoning emotional dependence on external events.",
        "key_concepts": "dichotomy of control, inner freedom, endurance, discipline of desire, emotional independence"
    },
    {
        "id": 17,
        "title": "On the Shortness of Life",
        "author": "Seneca",
        "school": "Stoicism",
        "summary": "An urgent essay arguing that life is sufficiently long if not wasted on pointless social ambitions, shallow pastimes, and endless procrastination.",
        "key_concepts": "wasted time, living deliberately, procrastination, mortality, present moment"
    },
    {
        "id": 18,
        "title": "Enchiridion",
        "author": "Epictetus",
        "school": "Stoicism",
        "summary": "A concise handbook of practical stoic rules to navigate daily stress, social expectations, and personal adversity without losing inner peace.",
        "key_concepts": "practical stoicism, cognitive reframing, mental armor, indifference to external praise"
    },

    # === 4. POLITICAL PHILOSOPHY & POWER ===
    {
        "id": 19,
        "title": "The Prince",
        "author": "Niccolò Machiavelli",
        "school": "Political Realism",
        "summary": "A blunt treatise analyzing how political leaders gain and keep power using pragmatic statecraft, strategic deception, and balancing fear with affection.",
        "key_concepts": "realpolitik, virtu vs fortuna, holding power, ends justify the means, pragmatic leadership"
    },
    {
        "id": 20,
        "title": "Leviathan",
        "author": "Thomas Hobbes",
        "school": "Social Contract",
        "summary": "Argues that without an absolute sovereign power, human life in the state of nature is solitary, poor, nasty, brutish, and short due to perpetual warfare.",
        "key_concepts": "state of nature, social contract, absolute sovereign, fear of violent death, political stability"
    },
    {
        "id": 21,
        "title": "The Social Contract",
        "author": "Jean-Jacques Rousseau",
        "school": "Political Enlightenment",
        "summary": "Proposes that legitimate political authority rests on a collective contract where citizens surrender natural liberties to enact the General Will.",
        "key_concepts": "general will, popular sovereignty, natural vs civil freedom, equality, legitimate authority"
    },
    {
        "id": 22,
        "title": "The Communist Manifesto",
        "author": "Karl Marx and Friedrich Engels",
        "school": "Marxism",
        "summary": "Analyzes human history as driven by class struggle, urging the working class to overthrow capitalist exploitation through organized revolution.",
        "key_concepts": "class struggle, historical materialism, proletariat, revolution, abolishing exploitation"
    },
    {
        "id": 23,
        "title": "Two Treatises of Government",
        "author": "John Locke",
        "school": "Classical Liberalism",
        "summary": "Defends fundamental natural rights to life, liberty, and property, arguing that governments exist solely by the consent of the governed.",
        "key_concepts": "natural rights, consent of the governed, private property, right of revolution, limited government"
    },

    # === 5. EASTERN PHILOSOPHY (CONFUCIANISM & TAOISM) ===
    {
        "id": 24,
        "title": "Mengzi",
        "author": "Mencius",
        "school": "Confucianism",
        "summary": "Defends the thesis that human nature is innately good, comparing morality to natural sprouts that flourish under benevolent governance and ethical education.",
        "key_concepts": "innate goodness of human nature, benevolent government, sprouts of virtue, mandate of heaven, righteous rebellion"
    },
    {
        "id": 25,
        "title": "Tao Te Ching",
        "author": "Lao Tzu",
        "school": "Taoism",
        "summary": "Foundational text promoting effortless action (wu wei), humility, simplicity, and living in complete harmony with the mysterious flow of the cosmos.",
        "key_concepts": "tao, wu wei, effortless action, humility, simplicity, softness overcoming hardness"
    },
    {
        "id": 26,
        "title": "Zhuangzi",
        "author": "Zhuang Zhou",
        "school": "Taoism",
        "summary": "Humorous allegories and paradoxes that question rigid intellectual categories, celebrate spontaneity, and explore freedom from societal expectations.",
        "key_concepts": "butterfly dream, spontaneity, radical relativism, usefulness of the useless, perspective transformation"
    },
    {
        "id": 27,
        "title": "The Analects",
        "author": "Confucius",
        "school": "Confucianism",
        "summary": "A collection of sayings emphasizing moral integrity, filial devotion, social harmony, and the lifelong cultivation of the exemplary gentleman (Junzi).",
        "key_concepts": "ren benevolence, filial piety, ritual propriety, junzi, social harmony, self-cultivation"
    },
    {
        "id": 28,
        "title": "Liezi",
        "author": "Lie Yukou",
        "school": "Taoism",
        "summary": "Philosophical fables illustrating quiet contemplation, non-attachment to fate or outcomes, and walking smoothly with natural changes.",
        "key_concepts": "spontaneity, non-striving, natural cycles, accepting fate, non-interference"
    },
    {
        "id": 29,
        "title": "The Great Learning",
        "author": "Confucius",
        "school": "Confucianism",
        "summary": "Outlines an organic path of moral progress starting with investigating things and purifying the self, expanding outward to pacifying the world.",
        "key_concepts": "self-cultivation, ordering the family, harmonious state, clear virtue, progressive morality"
    },
    {
        "id": 30,
        "title": "Xunzi",
        "author": "Xun Kuang",
        "school": "Confucianism",
        "summary": "Contrasts with Mencius by arguing that human nature is fundamentally undisciplined and selfish, requiring conscious ritual, education, and social law to achieve good.",
        "key_concepts": "human nature is raw, deliberate effort, ritual discipline, social order, moral transformation"
    }
]

df = pd.DataFrame(books_data)
df.to_csv(FILE_PATH, index=False, encoding="utf-8")
print(f"✅ Successfully generated dataset with {len(df)} books at: {FILE_PATH}")