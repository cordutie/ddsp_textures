---
layout: default
title: Texture Framework
width: 1100px
---

<div style="position: relative; width: 100%; height: 60px; margin-bottom: 20px;">
  <img src="assets/img/upf_logo.png" alt="UPF Logo"
       style="position: absolute; top: 50%; left: 0; max-height: 60px; max-width: 50%; height: auto; width: auto; transform: translateY(-50%);">
  <img src="assets/img/mtg_logo.png" alt="MTG Logo"
       style="position: absolute; top: 50%; right: 0; max-height: 60px; max-width: 50%; height: auto; width: auto; transform: translateY(-50%);">
</div>

<div style="text-align: center">
<h1>A Statistics-Driven Differentiable Approach for Sound Texture Synthesis and Analysis</h1>

<p>
  <a href="https://cordutie.github.io/"><strong>Esteban Gutiérrez</strong></a><sup>1</sup>, 
  <a href="https://ffont.github.io/"><strong>Frederic Font</strong></a><sup>1</sup>, 
  <strong>Xavier Serra</strong><sup>1</sup>, and  
  <a href="https://lonce.org/"><strong>Lonce Wyse</strong></a><sup>1</sup>
</p>

<p><sup>1</sup> <em>Department of Information and Communications Technologies, Universitat Pompeu Fabra</em></p>
</div>

<div style="text-align: center; margin-top: 1em; margin-bottom: -0.3em;">
  <a href="https://doi.org/10.48550/arXiv.2506.04073" 
    style="display: inline-block; background: #3498db; color: white; padding: 0.6em 1em; border-radius: 5px; text-decoration: none; margin: 0.3em 0em;">
    📄 Paper
  </a>
  <a href="https://github.com/cordutie/ddsp_textures" 
    style="display: inline-block; background: #3498db; color: white; padding: 0.6em 1em; border-radius: 5px; text-decoration: none; margin: 0.3em 0em;">
    <img src="/assets/img/gh_logo.png" alt="GitHub" width="15" style="filter: invert(1); vertical-align: -0.2em;" /> <Code style="color: white; font-size: 0.85em;">TexDSP</Code> repository
  </a>
  <a href="https://github.com/cordutie/texstat" 
    style="display: inline-block; background: #3498db; color: white; padding: 0.6em 1em; border-radius: 5px; text-decoration: none; margin: 0.3em 0em;">
    <img src="/assets/img/gh_logo.png" alt="GitHub" width="15" style="filter: invert(1); vertical-align: -0.2em;" /> <Code style="color: white; font-size: 0.85em;">TexStat</Code> repository
  </a>
  <!-- <a href="#"
    style="display: inline-block; background: #3498db; color: white; padding: 0.6em 1em; border-radius: 5px; text-decoration: none; margin: 0.3em 0em;">
    💻 Blog post
  </a> -->
  <a href="#" onclick="event.preventDefault(); navigator.clipboard.writeText('@inproceedings{gutierrez2025statistics,\n title     = {A Statistics-Driven Differentiable Approach for Sound Texture Synthesis and Analysis},\n author    = {Esteban Gutiérrez and Frederic Font and Xavier Serra and Lonce Wyse},\n booktitle = {Proceedings of the 28th International Conference on Digital Audio Effects (DAFx25)},\n year      = {2025},\n address   = {Ancona, Italy},\n month     = {September},\n note      = {2--5 September 2025}\n}'); alert('Copied to clipboard!');"
    style="display: inline-block; background: #3498db; color: white; padding: 0.6em 1em; border-radius: 5px; text-decoration: none; margin: 0.3em 0em;">
    📚 Bibtex
  </a>
</div>

<div style="margin-top: 20px;"></div>
<p>
This webpage provides supplementary materials for our paper <em>"A Statistics-Driven Differentiable Approach for Sound Texture Synthesis and Analysis"</em> to be presented at the 25th edition of the Digital Audio Effects (DAFx) Conference in Ancona, Italy.
</p>

<div style="margin-top: 20px;"></div>
<h2><strong>1. Introduction</strong></h2>
{% include_relative 1_introduction.md %}

<div style="margin-top: 40px;"></div>
<h2><strong>2. Models</strong></h2>
{% include_relative 2_models.md %}

<div style="margin-top: 40px;"></div>
<h2><strong>3. Experiments and Sound Examples</strong></h2>
{% include_relative 3_experiments.md %}

<div style="margin-top: 40px;"></div>
<div style="display: flex; justify-content: center;">
  <div style="border-left: 4px solid rgb(200, 200, 200); background:rgb(230, 230, 230); padding: 1em 1.2em; margin: 1.5em 0; border-radius: 8px; max-width: 300px; width: 100%;">
    <strong>Legend:</strong>
    <ul style="list-style: none; padding: 0.5em 0 0 0; margin: 0;">
      <li>🎧 Sound examples included</li>
      <li>📊 Numerical experiments included</li>
      <li>📖 Theory included</li>
      <li>🚧 Still under construction</li>
    </ul>
  </div>
</div>

<h2><strong>Acknowledgements</strong></h2>

This work has been supported by the project "IA y Música: Cátedra en Inteligencia Artificial y Música (TSI-100929-2023-1)", funded by the "Secretaría de Estado de Digitalización e Inteligencia Artificial and the Unión Europea-Next Generation EU".

<div style="justify-content: center; width: 100%;">
  <img src="assets/img/chair.png" alt="Funding" style="top: 50%; right: 0; width: 100%;">
</div>
