---
title: Why I Ditched Modern IDEs for Neovim (And What It Taught Me About Coding)
date: 2025-01-22T12:00:00-05:00
tags:
  - neovim-experience
  - neovim
  - nvim
author: Sajad
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
description: How did uninstalling vscode and installing neovim go
disableHLJS: false
disableShare: false
hideSummary: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
cover:
  image: /images/neovim.png
  alt: Neovim
  relative: false
  hidden: false
---

Six months ago, I made a decision that felt borderline reckless: I uninstalled VS Code, my trusty code companion for five years, and switched to Neovim. In other words I did `rm -rf vscode; nvim .`

I'm a full time SWE working on projects with different languages and setups, So this was not just a casual experiment, It was a hard pivot that suddenly reduced my productivity. I spent weekends and after-work time to tweak configs, Figuring out Lua syntax and muttering curses LSPs. I tried and I could adopt with it, It taught me great lessons about coding efficiency, tooling, project structure, file browsing and many more.

Here is why I don't regret it and what I wish I'd known sooner.

---

### **First Month: “What Have I Done?”**  
TBH My first week felt like learning to code all over again.  

- **Lua** I’d never written a line of Lua. Suddenly, I was knee deep in `init.lua`, figuring out with plugin managers (`Lazy.nvim`, I love you now) and debugging cryptic errors.  
- **Keybindings:** It was pretty hard to remember keybinding and using them in my muscle memory, I had to always write them down on a paper to remember them. Don't ask me about switching from CMD+S to `:wq` , It was really hard. The other problem was copy pasting, You basically cannot copy paste into and out of Neovim without selecting the correct register and it takes time to take place in muscle memory. 
- **Goodbye dear plugins:** I replaced VS Code’s luxuries ESLint, Prettier, IntelliSense with `nvim-lsp`, `treesitter`, and `mason`.

My productivity reduced. I worked slower and I was figuring out the tool instead of the problem. But I refused to revert it, because something kept nagging me: *What if this pain is teaching me more than comfort ever did?*  

---

### **The Turning Point: “Oh. *This* Is Why People Do This.”**  
Around month two, things clicked. Neovim stopped fighting me and I was starting to enjoy it.

- **Speed Is a Feature:** No more mouse. I could simply navigate between files with `Telescope`, jump between splits with `Ctrl+hjkl`, and bulk change code with `%s` substitution.
- **Understanding the Machine:** Configuring my own LSP setup for TypeScript and other langs gave me a good grasp of how these things work out of the box. I *finally* grasped how linting, formatting, and autocompletion *actually work* under the hood, knowledge that’s made me a better debugger.  
- **Minimal and simplictiy:** Modern IDEs bombard you with features. Neovim forced me to ask: *What do I **truly** need?*, I could add only things I need and keep my focus on the code instead of thousands of icons, Texts and notifications. 

---

### **What Neovim Taught Me About Coding** 
Beyond the editor wars, this experiment reshaped how I approach coding work:

1. Modern tools prioritize user-friendliness over customization options. Neovim taught me to tailor workflows to my needs—not the other way around.
2. Using CLI tools like `git` and `npm` directly within Neovim's terminal enhanced my workflow. I now write bash scripts for repetitive tasks instead of relying on GUI buttons.
3. **Minimalism Scales:** A lightweight editor forces you to write cleaner code by removing all the distractions. I can now think harder about naming, structure, and patterns. My components got smaller, my state logic more efficient, and my reliance on frameworks and libraries more mindful.

---

Is Neovim the right editor for you? Neovim might not be for everyone. If you’re on a tight deadline or value simplicity, stick with VS Code. Want to understand your tools better and regain control of your craft? Give it a shot.

Yes, the learning curve is brutal. Don't be surprised if you want to quit. But surviving that gauntlet will make you a sharper, more resourceful developer.

Six months later, I’m coding faster, thinking deeper, and ironically appreciating modern IDEs more. Because now, I *choose* my tools. They don’t choose me.

Neovim's interface felt like a step back from the ease of a self-driving car. Driving a manual transmission taught me more about coding than any IDE ever did.