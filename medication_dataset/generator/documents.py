"""
Document builders: turn one medication's facts into realistic documents of 12 types.

Every document is about exactly ONE medication. That is what keeps the dataset modular: all text about a
medication lives in that medication's files, so a medication can be added or removed without touching others.

bank = patterns.TRAIN (training / validation corpus) or patterns.HELDOUT (test corpus).
"""
from .patterns import FACT_ATTRS, FILLER, ORG, DESK, STAFF
from .questions import Q


def fields(m):
    c, s = m["conditions"], m["side_effects"]
    return {"name": m["name"], "cond1": c[0], "cond2": c[1], "se1": s[0], "se2": s[1],
            "shelf": f"{m['shelf_life_months']} months", "price": f"${m['price_usd']:.2f}",
            "forms": m["form_phrase"], "form": m["form"], "org": ORG, "desk": DESK}


def cap(s):
    return s[0].upper() + s[1:] if s else s


def fill(t, f):
    return cap(t.format(**f))


class Writer:
    """Holds one medication, a random generator and a pattern bank; produces sentences."""

    def __init__(self, m, rng, bank):
        self.m, self.rng, self.bank, self.f = m, rng, bank, fields(m)

    def say(self, attr):
        return fill(self.rng.choice(self.bank[attr]), self.f)

    def filler(self, kind="generic", k=1):
        return [fill(t, self.f) for t in self.rng.sample(FILLER[kind], k)]

    def attrs(self, lo=3, hi=6, must=()):
        a = list(must) + [x for x in self.rng.sample(FACT_ATTRS, len(FACT_ATTRS)) if x not in must]
        return a[:self.rng.randint(max(lo, len(must)), hi)]

    def staff(self):
        return self.rng.choice(STAFF)

    def para(self, sents):
        return " ".join(sents)


# ------------------------------------------------------------------------------------------------ builders
def leaflet(w):
    n = w.f["name"]
    title = w.rng.choice([f"Patient Information Leaflet: {n}", f"{n}: Information for the Patient",
                          f"Package leaflet for {n}"])
    sections = [
        (["treats", "form"], [f"What {n} is and what it is used for", f"About {n}", f"What is {n}?"]),
        (["side_effects", "safety"], ["Possible side effects", "Side effects", f"Side effects of {n}"]),
        (["dosage"], [f"How to take {n}", "Dosage", f"Using {n}"]),
        (["shelf_life", "safety"], [f"How to store {n}", "Storage and expiry", f"Keeping {n}"]),
        (["price"], ["Price", "Cost", "Price information"]),
    ]
    keep = sorted(w.rng.sample(range(5), w.rng.randint(3, 5)))
    out, used = [title, ""], []
    for i, k in enumerate(keep):
        attrs, heads = sections[k]
        sents = [w.say(a) for a in attrs if a != "safety" or w.rng.random() < 0.6]
        used += [a for a in attrs if a != "safety"]
        out += [f"{i + 1}. {w.rng.choice(heads)}", w.para(sents), ""]
    out.append(w.filler()[0])
    return "\n".join(out), used


def faq(w):
    n = w.f["name"]
    attrs = w.attrs(3, 6)
    out = [w.rng.choice([f"Frequently asked questions about {n}", f"{n}: your questions answered", f"FAQ: {n}"]), ""]
    for a in attrs:
        out += ["Q: " + fill(w.rng.choice(Q[a]["train"]), w.f), "A: " + w.say(a), ""]
    out.append(w.filler()[0])
    return "\n".join(out), attrs


def chat(w):
    n = w.f["name"]
    attrs = w.attrs(2, 4)
    agent = w.staff()[0].split()[0]
    out = [f"Chat transcript, {DESK}", "",
           f"Agent ({agent}): Hello, you're through to the {DESK} of {ORG}. How can I help you today?",
           f"Customer: Hi, I have some questions about {n}.",
           f"Agent ({agent}): " + w.rng.choice(["Of course. What would you like to know?", "Sure, go ahead.",
                                                "Happy to help. What is your question?"])]
    for i, a in enumerate(attrs):
        q = fill(w.rng.choice(Q[a]["train"]), w.f)
        if i:
            lead = w.rng.choice(["Also, ", "One more thing. ", "Thanks. ", "Okay. "])
            if lead == "Also, " and not q.startswith(n):
                q = q[0].lower() + q[1:]
            q = lead + q
        out += [f"Customer: {q}", f"Agent ({agent}): {w.say(a)}"]
    out += ["Customer: That's all, thank you.", f"Agent ({agent}): You're welcome. {w.filler('warm')[0]}"]
    return "\n".join(out), attrs


def internal_email(w):
    n = w.f["name"]
    who, role = w.staff()
    attrs = w.attrs(2, 5)
    subj = w.rng.choice([f"Product reminder: {n}", f"Customer questions about {n}", f"Quick notes on {n}",
                         f"{n} - key facts for the desk"])
    body = [w.filler("internal")[0]] + [w.say(a) for a in attrs] + w.filler("internal")
    return "\n".join([f"From: {who} ({role})", f"To: {DESK} team", f"Subject: {subj}", "",
                      w.rng.choice(["Hi team,", "Hello everyone,", "Team,"]), "", w.para(body), "",
                      w.rng.choice(["Thanks,", "Best,", "Regards,"]), who]), attrs


def customer_notice(w):
    n = w.f["name"]
    attrs = w.attrs(2, 4)
    title = w.rng.choice([f"Customer notice: {n}", f"Important information for {n} customers",
                          f"Notice to customers who use {n}"])
    body = [w.say(a) for a in attrs] + ([w.say("safety")] if w.rng.random() < 0.5 else []) + w.filler()
    return "\n".join([title, "", "Dear customer,", "", w.para(body), "", f"The {DESK}", ORG]), attrs


def price_list(w):
    f = w.f
    rows = [("Form", cap(f["form"]), "form"), ("Pack price", f["price"], "price"),
            ("Shelf life", f"{f['shelf']} from manufacture", "shelf_life"),
            ("Used for", f"{f['cond1']}, {f['cond2']}", "treats"),
            ("Common side effects", f"{f['se1']}, {f['se2']}", "side_effects"),
            ("Dosage", "ask a doctor or pharmacist", "dosage")]
    keep = [r for r in rows if w.rng.random() < 0.75] or rows[:2]
    out = [w.rng.choice(["Product catalog entry", f"{ORG} price list", "Catalog listing"]), f"Product: {f['name']}"]
    out += [f"{k}: {v}" for k, v, _ in keep]
    used = [a for _, _, a in keep]
    extra = w.rng.choice(["price", "shelf_life", "form"])
    out += ["", w.say(extra)]
    return "\n".join(out), sorted(set(used + [extra]))


def inventory_report(w):
    n = w.f["name"]
    attrs = w.attrs(2, 4, must=["shelf_life"])
    out = [w.rng.choice([f"Warehouse stock report: {n}", f"Inventory check: {n}", f"Stock review for {n}"]),
           f"Prepared by {STAFF[3][0]}, {STAFF[3][1]}", "",
           w.para([w.rng.choice(["All stock was checked against the expiry dates this week.",
                                 "This report covers the stock held in the main warehouse.",
                                 "Expired stock is removed from the shelves at every check."])]
                  + [w.say(a) for a in attrs]
                  + [w.rng.choice(["No damaged packs were found.", "Stock levels are normal.",
                                   "Older packs are shipped first."])])]
    return "\n".join(out), attrs


def training_guide(w):
    n = w.f["name"]
    attrs = w.attrs(3, 6, must=["dosage"])
    lead = {"treats": "what the medicine is for", "side_effects": "side effects", "shelf_life": "how long it lasts",
            "price": "the price", "form": "what form it comes in", "dosage": "the dose"}
    out = [w.rng.choice([f"Agent training guide: questions about {n}", f"Training module: {n}",
                         f"How to answer customers about {n}"]), f"Written by {STAFF[2][0]}, {STAFF[2][1]}", ""]
    for a in attrs:
        if a == "dosage":
            out.append(f"If a customer asks about the dose, never give an amount. Say: \"{w.say('dosage')}\"")
        else:
            out.append(f"If a customer asks about {lead[a]}, you can say: \"{w.say(a)}\"")
    out += ["", w.filler("internal")[0]]
    return "\n".join(out), attrs


def safety_bulletin(w):
    n = w.f["name"]
    attrs = w.attrs(1, 3, must=["side_effects", "dosage"])
    body = [w.say(a) for a in attrs] + [w.say("safety") for _ in range(w.rng.randint(1, 2))]
    return "\n".join([w.rng.choice([f"Safety bulletin: {n}", f"Safety information for {n}", f"Using {n} safely"]),
                      "", w.para(body), "", w.filler()[0]]), attrs


def reply_letter(w):
    n = w.f["name"]
    who, role = w.staff()
    attrs = w.attrs(1, 3)
    body = [f"Thank you for contacting the {DESK} about {n}."] + [w.say(a) for a in attrs] + w.filler("warm")
    return "\n".join(["Dear customer,", "", w.para(body), "", "Kind regards,", f"{who}, {role}", ORG]), attrs


def newsletter(w):
    n = w.f["name"]
    attrs = w.attrs(2, 5)
    body = [w.rng.choice([f"This month we take a closer look at {n}.", f"In this issue we focus on {n}.",
                          f"Many of you have asked us about {n}."])] + [w.say(a) for a in attrs] + w.filler()
    return "\n".join([f"{ORG} Customer Newsletter", "", w.rng.choice([f"Product spotlight: {n}", f"Spotlight on {n}"]),
                      "", w.para(body)]), attrs


def product_label(w):
    f = w.f
    lines = [f"{f['name']} by {ORG}", cap(f["forms"]), f"Uses: {f['cond1']} and {f['cond2']}",
             f"Possible side effects: {f['se1']}, {f['se2']}", f"Shelf life: {f['shelf']} from manufacture",
             f"Price: {f['price']} per pack", "Dosage: ask your doctor or pharmacist",
             "Keep out of the reach of children."]
    return "\n".join(lines), ["form", "treats", "side_effects", "shelf_life", "price", "dosage"]


# name -> (builder, weight in training/validation corpus, used in test corpus?)
TYPES = {
    "patient_leaflet": (leaflet, 1.0, True),
    "faq": (faq, 1.0, False),                       # contains training questions -> not in test
    "chat_transcript": (chat, 1.0, False),          # contains training questions -> not in test
    "internal_email": (internal_email, 1.0, True),
    "customer_notice": (customer_notice, 1.0, True),
    "price_list": (price_list, 0.5, False),         # fixed field layout, not a "wording" -> not in test
    "inventory_report": (inventory_report, 0.7, True),
    "training_guide": (training_guide, 1.0, True),
    "safety_bulletin": (safety_bulletin, 0.8, True),
    "reply_letter": (reply_letter, 1.0, True),
    "newsletter": (newsletter, 0.8, True),
    "product_label": (product_label, 0.3, False),   # fixed field layout -> not in test
}


def make_doc(m, rng, bank, test=False):
    names = [k for k, v in TYPES.items() if v[2] or not test]
    weights = [1.0 if test else TYPES[k][1] for k in names]
    kind = rng.choices(names, weights)[0]
    text, used = TYPES[kind][0](Writer(m, rng, bank))
    return {"type": kind, "facts": sorted(set(used)), "text": text.strip()}
