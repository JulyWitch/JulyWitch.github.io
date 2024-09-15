---
title: "3 Necessarily VS Code extensions for a developer"
date: 2017-03-02T12:00:00-05:00
tags: ["vscode-plugins", "vscode"]
author: "Sajad"
showToc: true
TocOpen: false
draft: false
hidemeta: false
comments: true
description: "Add more functionalities to your VS Code"
disableHLJS: true # to disable highlightjs
disableShare: false
disableHLJS: false
hideSummary: false
searchHidden: true
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
cover:
    image: "/images/posts/vscode-plugins/vscode.png" # image path/url
    alt: "VS Code" # alt text
    relative: false # when using page bundles set this to true
    hidden: false # only hide on current single page
---

VS Code is a popular code editor among developers.

Using VS Code extensions, We have the chance to add more functionalities to this editor than just being a code editor.

If you are using VS Code you probably already have installed common extensions like Git and Prettier so I will skip them.

## 1. Git Graph

This is a helpful extension that lets you see the Git history in a fancy way. Git’s desktop app already 
has this feature but you will need an extension to use it in VS Code.

All you need to do is to [install it](https://marketplace.visualstudio.com/items?itemName=mhutchie.git-graph) and then navigate to a git project and click on Git Graph text on the bottom bar of the VS Code.


Here you can see the Git Graph of Flutter repo (Blue line is the main branch)

{{< figure src="/images/posts/vscode-plugins/gitgraph.png" title="Git Graph" >}}

## 2. VS Code Google Translate
I personally love this one, It translates raw texts in VS Code using a simple, fast, and free API.


After [installing it](https://marketplace.visualstudio.com/items?itemName=funkyremi.vscode-google-translate), Go to a file and select a text, Then open the command palette and type translate, 
Select `Transalte selection(s)` and select the target language.

It will help you when you are using a foreign API, Or when you want a simple translation for your app.

## 3. WakaTime

This one is actually a work time tracking extension and it tracks your coding time, 
You need to register on [WakaTime](https://wakatime.com/) and then [install the extension](https://wakatime.com/vs-code) on VS Code.
After providing the API Key from its website, It will start working and sends data to your dashboard](https://wakatime.com/dashboard)


{{< figure src="/images/posts/vscode-plugins/wakatime.png" title="WakaTime" >}}
