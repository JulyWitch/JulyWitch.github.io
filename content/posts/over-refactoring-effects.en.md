---
title: "The Over-Refactoring Trap: When to Refactor and When to Stop"
date: 2025-01-25T12:00:00-05:00
tags:
  - refactor
  - over-refactor
  - refactor-hell
  - dry vs lob
author: Sajad
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
description:
disableHLJS: false
disableShare: false
hideSummary: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
cover:
  image: /images/refactor-yes-but.png
  alt: YesBut
  relative: false
  hidden: false
---

In the early days of my career (around 5 years ago), I was tasked with building a simple social media mobile app in Flutter.
I developed the application to MVP version, and along the way, I refactored every single widget with the intention
of acheiveing "clean code".
I was obsessed with DRY principle and created countless reusable code blocks, all to avoid rewriting the same code.

But I'd already made my first mistake

> Too much refactoring doesn't make the code cleaner, It just makes it harder, hard to navigate, hard to change and hard to understand.

### The Consequences

A few weeks later, the consequences hit hard. A tester reported a bug in phone number input of login page,
I was confident in my "clean" codebase, I jumped in, fixed the bug in my reusable `PhoneNumberInput` component,
tested the login page, and pushed the update.

But then the chaos began,

I was using `PhoneNumberInput` UI component on a handful number of pages, login page, profile page, add user page and invite page,
Initially it meant to be just a simple input but
fixing the bug in login page triggered a domino effect and introduced a new bug in profile page. As the time passed,
I kept adding functionalities to the simple input until it became a big chunk of logic and state-management and UI.

```jsx

function PhoneNumberInput({ initialValue, checkUniqueness, countriesList, checkIcon, debounceDuration, ...}) {}

```

Suddenly I saw myself taking care of a big component that is not abstract anymore.
I admit it, I was wrong and in a rapid develoment to reach to the deadline,
ant there was no time for breaking countless reusable but complicated and error prone components.

---

Funny how it gets you into an infinite loop of debugging and fixing. The tasks that were meant to take half an hour now take a full day.

The actual problem wasn’t how I wrote the `PhoneNumberInput` or how I abstracted it. The problem was its very existence.
I refactored the wrong component.

### The Reality of Abstractions

At the start of a product, everything is easy, and Our abstractions seem logical and elegant. **Everything Makes Sense**.

But as the business grows, things change.

PM, testers, and client feedback start shaping the product,
pushing it beyond the limits you unintentionally set with your refactoring.

### The Illusional Feeling

Refactoring feels good, Over-Refactoring feels even better,
In those days, I kept telling myself "Oh, I'm improving the Code",
but each bug, each edge case, and each feedback was ruining my abstractions
and making the underlying code complex.

The other problem is onboarding new members, When
using the known components and refactoring wisely, new members can join the team
and start working quickly, but with over-refactored components,
They face a steep learning curve.

### Wrapping It Up

Hard truths I learned:

- Abstraction you make are perfect at the start but will be complex as the product grows
- Over-Refactoring causes domino effect and leads to unintended bugs
- The onboarding of new team members takes longer when code is over-refactored
- The development speed decreases, While you might avoid duplicating code, you still
  need to check multiple functionalities when you make a simple change in one.
- Too much refactoring doesn't make the code cleaner, It just makes it "look" clean

**Surface Beauty, But Hidden Chaos**.

![YesBut](/images/refactor-yes-but.png)

### Where to refactor

My rules for refactoring are pretty simple

- The code block is repeated enough
- The code block and what it does is atomic
- It is a well defined task

### When to stop

- **"Oh I'm improving the Code":** Refactoring makes us developers feel productive,
  Now your duty is to find out if this refactoring is gonna add user value or is it just the "Programmer Ego" telling you to do this.
- **The lazy programmer mindset:** Refactoring 3 blocks of code takes more time than changing them one by one, So why even bother?
- **Adding more params:** If you find yourself adding new params to your already refactored code block, It's time to rethink your approach

What’s your experience with over-refactoring? Have you ever encountered a codebase like this?
