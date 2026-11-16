"""
Sentence patterns: the different wordings used to state each kind of fact.

Placeholders (filled from a medication's knowledge-base file):
  {name}   medication name                      {forms}  form with article, e.g. "a nasal spray", "tablets"
  {cond1}, {cond2}  the two conditions it treats    {se1}, {se2}  its two common side effects
  {shelf}  shelf life, e.g. "36 months"           {price}  price of one pack, e.g. "$12.40"
  {org}    company name                           {desk}   department name

TRAIN patterns are used in the training and validation corpus.
HELDOUT patterns are used ONLY in the test corpus (and the "unseen" probes), so the test set measures
whether the model can handle wordings it has never seen.

Dosage rule of the dataset: no document ever states a dose. Every dosage or safety-of-use statement sends the
reader to a doctor or pharmacist.
"""

ORG = "Halverin Medical Logistics"
DESK = "Medication Support Desk"

TRAIN = {
    "treats": [
        "{name} is used to treat {cond1} and {cond2}.",
        "Doctors recommend {name} for {cond1} and {cond2}.",
        "{name} helps relieve {cond1} as well as {cond2}.",
        "If you have {cond1} or {cond2}, {name} may be suggested for you.",
        "{name} is a treatment for {cond1} and {cond2}.",
        "The approved uses of {name} are {cond1} and {cond2}.",
        "Patients take {name} to manage {cond1} and {cond2}.",
        "{name} works against {cond1} and also against {cond2}.",
        "Most of our customers use {name} for {cond1} and {cond2}.",
        "{name} was developed for people dealing with {cond1} or {cond2}.",
        "What is {name} for? It treats {cond1} and {cond2}.",
        "{name} is indicated for {cond1} and {cond2}.",
    ],
    "side_effects": [
        "Common side effects of {name} include {se1} and {se2}.",
        "{name} can cause {se1} and {se2} in some people.",
        "Some patients notice {se1} or {se2} after using {name}.",
        "The most frequently reported side effects of {name} are {se1} and {se2}.",
        "While using {name}, you may experience {se1} and {se2}.",
        "{se1} and {se2} are known side effects of {name}.",
        "Possible side effects of {name} are {se1} and {se2}.",
        "{name} may lead to {se1} and {se2}.",
        "Watch for {se1} and {se2} when you start {name}.",
        "The side effects people mention most with {name} are {se1} and {se2}.",
        "A few users of {name} report {se1} and {se2}.",
        "With {name}, the usual side effects are {se1} and {se2}.",
    ],
    "shelf_life": [
        "{name} has a shelf life of {shelf}.",
        "Unopened {name} stays good for {shelf} from the date of manufacture.",
        "The shelf life of {name} is {shelf}.",
        "{name} can be stored for {shelf} before it expires.",
        "{name} expires {shelf} after it is made.",
        "Each pack of {name} is good for {shelf}.",
        "Stock of {name} remains usable for {shelf} after manufacture.",
        "Do not use {name} more than {shelf} after its manufacturing date.",
        "{name} keeps for {shelf} when stored properly.",
        "From the manufacturing date, {name} lasts {shelf}.",
        "Our warehouse rotates {name} stock because its shelf life is {shelf}.",
        "The expiry period for {name} is {shelf}.",
    ],
    "price": [
        "A pack of {name} costs {price}.",
        "{name} is priced at {price} per pack.",
        "The price of {name} is {price} for one pack.",
        "You can buy a pack of {name} for {price}.",
        "{name} sells for {price} a pack.",
        "One pack of {name} is {price}.",
        "The current price for {name} is {price} per pack.",
        "Customers pay {price} for each pack of {name}.",
        "{name} costs {price} per pack.",
        "Our listed price for {name} is {price} a pack.",
        "Each pack of {name} is sold at {price}.",
        "{name} retails at {price} per pack.",
    ],
    "form": [
        "{name} comes as {forms}.",
        "{name} is supplied as {forms}.",
        "{name} is available as {forms}.",
        "You will receive {name} in the form of {forms}.",
        "{name} is made as {forms}.",
        "{name} is sold as {forms}.",
        "We ship {name} as {forms}.",
        "{name} is produced as {forms}.",
        "Customers get {name} as {forms}.",
        "{name} is offered as {forms}.",
    ],
    "dosage": [
        "For the right dose of {name}, please talk to your doctor or pharmacist.",
        "We cannot give dosage advice for {name}; your doctor or pharmacist will tell you how much to take.",
        "How much {name} to take depends on you, so ask your doctor or pharmacist.",
        "Always follow your doctor's or pharmacist's instructions for the dose of {name}.",
        "Dosage questions about {name} should go to your doctor or pharmacist.",
        "Your doctor or pharmacist is the right person to decide your dose of {name}.",
        "Do not change your dose of {name} without talking to your doctor.",
        "Our support team does not give dosing instructions for {name}. Please speak with your doctor or pharmacist.",
        "Only your doctor or pharmacist can tell you the correct amount of {name} for you.",
        "Check with your doctor or pharmacist before taking {name}, especially about the dose.",
    ],
    # general safety lines (the same for every medication; they never contain a dose)
    "safety": [
        "If you take more {name} than you should, contact a doctor or emergency services right away.",
        "Get medical help immediately if you have trouble breathing or swelling of the face after using {name}.",
        "Keep {name} out of the reach of children.",
        "Store {name} in a cool, dry place away from direct sunlight.",
        "Tell your doctor about any other medicines you take before starting {name}.",
        "Ask your doctor or pharmacist before using {name} if you are pregnant or breastfeeding.",
        "Do not use {name} after the expiry date printed on the pack.",
        "If a side effect becomes serious or does not go away, stop using {name} and call your doctor.",
    ],
}

HELDOUT = {
    "treats": [
        "People struggling with {cond1} or {cond2} often turn to {name}.",
        "{name} targets two conditions: {cond1} and {cond2}.",
        "The conditions {name} is meant for are {cond1} and {cond2}.",
    ],
    "side_effects": [
        "Taking {name} sometimes brings on {se1} and {se2}.",
        "Be prepared for the possibility of {se1} and {se2} with {name}.",
        "Two reactions linked to {name} are {se1} and {se2}.",
    ],
    "shelf_life": [
        "Once made, {name} remains fit for use for {shelf}.",
        "A batch of {name} reaches its expiry date {shelf} after production.",
        "Plan on {name} staying usable for {shelf}.",
    ],
    "price": [
        "Expect to spend {price} on a pack of {name}.",
        "Buying {name} sets you back {price} per pack.",
        "The going rate for a pack of {name} is {price}.",
    ],
    "form": [
        "{name} reaches customers as {forms}.",
        "In our catalog, {name} is listed as {forms}.",
        "{name} is packaged as {forms}.",
    ],
    "dosage": [
        "Please leave decisions about how much {name} to use to your doctor or pharmacist.",
        "The amount of {name} you should take is something to discuss with your doctor or pharmacist.",
        "Speak to a doctor or pharmacist to learn the correct dose of {name}.",
    ],
    "safety": [
        "Should you swallow too much {name}, reach a doctor or emergency services without delay.",
        "Children must not be able to get hold of {name}.",
        "Keep {name} somewhere cool and dry, out of strong sunlight.",
    ],
}

FACT_ATTRS = ["treats", "side_effects", "shelf_life", "price", "form", "dosage"]

# Background staff of the desk (fixed roles; they never state facts of their own)
STAFF = [("Dana Whitfield", "Support Lead"), ("Marcus Ode", "Pharmacy Liaison"),
         ("Priya Sen", "Training Coordinator"), ("Tom Alvarez", "Warehouse Manager")]

# Sentences without any medication fact, used to make documents read naturally
FILLER = {
    "generic": [
        "Thank you for choosing {org}.",
        "Please keep this information for future reference.",
        "Our team is here to help with any questions about your order.",
        "This information is provided by the {desk} of {org}.",
        "We want every customer to understand the medicines we deliver.",
        "If anything is unclear, contact the {desk}.",
        "We review our product information regularly to keep it accurate.",
        "Read the whole leaflet before you start using the medicine.",
        "Customer safety is our first priority.",
        "We are happy to answer general questions about our products.",
    ],
    "internal": [
        "Please share this with anyone covering the phones this week.",
        "We have had several customer questions about this product recently.",
        "Make sure your answers match the product sheet.",
        "Remember that we never give dosing advice on the phone or in chat.",
        "Thanks for keeping our answers consistent.",
        "Let me know if you notice any errors in the product sheets.",
        "This will also be covered at the next team meeting.",
        "Please log every customer question in the support system.",
    ],
    "warm": [
        "We hope you feel better soon.",
        "We know that health questions can be stressful.",
        "Your wellbeing matters to us.",
        "We appreciate you reaching out.",
        "It is always a good idea to ask when you are unsure.",
    ],
}
