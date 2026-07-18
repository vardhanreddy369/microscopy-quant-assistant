# Three-minute demonstration script

The software is the ticket, not the show. Its whole job is to earn the right to
ask the last question on this page.

## Before the meeting

- [ ] `streamlit run app.py` once, on the machine and screen you will use.
- [ ] Confirm it opens on **Human mitosis (real, fluorescence)** with a result
      already on screen. No clicking required to show something.
- [ ] Screen-record the full run as a backup, in case the laptop misbehaves.
- [ ] Screenshot the annotated image and the measurement table as a second backup.
- [ ] Close every other window. Increase the display font size.
- [ ] Confirm it works with Wi-Fi off. It should; nothing needs a network.

## Opening (about 20 seconds)

> "Professor, I wanted to show you something small rather than only describe my
> technical skills. I built this over a couple of days using a public microscopy
> image. It's meant to demonstrate how I can turn biological images into
> reproducible quantitative data."

Do not oversell. "A couple of days" is the point: it signals what a short
conversation about a real need could turn into.

## The run (about 90 seconds)

1. The app is already showing the real fluorescence image, analysed.
2. Point at the count: **284 objects detected**.
3. Zoom the annotated panel. Point at a pair of touching nuclei with a line
   between them.
   > "Those two nuclei are touching. A simple threshold counts them as one
   > object. This separates them, which is where most of the counting error
   > usually comes from."
4. Open the measurement table. Scroll once.
   > "Every object has area, shape, and signal intensity."
5. Click **Measurements (CSV)**. Let the file land.
   > "That's the part that matters: it comes out as a spreadsheet, the same way
   > every time."

## Volunteer the limitation (about 30 seconds)

Do this **before** he asks. It is the most credible thing you will say.

Switch to **Synthetic - difficult (known failure case)**.

> "I also want to show you where it fails. This image is dense and overlapping,
> and it finds about 72 of the 110 objects that are actually there. I know that
> number because I generated the image, so I know the truth. The method can't
> separate objects that overlap past a certain point, and no amount of tuning
> fixes it. I'd rather show you that than pretend it always works."

## The question (about 20 seconds)

This is the actual purpose of the meeting.

> "I used public data, so this isn't tailored to your lab at all. What I really
> wanted to ask is whether your students have any image-analysis or repetitive
> data-analysis work where something like this could be adapted."

Then stop talking and let him answer.

## If he asks "how accurate is it?"

> "On images where I know the true count, it's exact on separated and touching
> nuclei, and about 65% on dense overlapping ones. On your images I have no
> idea, and I wouldn't claim otherwise until I'd tested it against slides
> someone in the lab had annotated by hand."

## If he asks "we already use ImageJ / Fiji"

Do not argue. That is the right answer for most labs.

> "That makes sense, and for a one-off measurement Fiji is hard to beat. Where
> something custom tends to help is when the same analysis runs over hundreds of
> images and has to come out identical every time, or when the output needs to
> land in a specific format. If that's not a pain point, I'd rather hear what
> actually is."

## If he mentions sequencing rather than imaging

> "Then the image part is the wrong half. The same interface works for
> tabular data. What does that analysis look like now?"

## Within 48 hours of the meeting

Send a short email, no more than two paragraphs:

1. One sentence thanking him and naming the specific thing he said.
2. One concrete proposal tied to that thing, scoped to about two weeks.
3. One ask: a handful of representative images, or a 20-minute call with
   whichever student owns the workflow.

The meeting converts on this email, not on the demo.
