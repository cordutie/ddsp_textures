
<p>
In this work we introduce <code>TexStat</code>, a perceptually grounded loss function inspired by McDermott and Simoncelli’s work. Alongside it, we present <code>TexEnv</code>, a lightweight differentiable synthesizer, and <code>TexDSP</code>, a DDSP-style generative model tailored for texture audio. All tools are open-source, implemented in PyTorch, and designed for efficient training and evaluation. Below, a small set of highlighted examples generated with <code>TexDSP</code> can be found.
</p>

<div style="overflow-x: auto; max-width: 80%; margin: 0 auto; padding: 10px; box-sizing: border-box;">
  <div style="text-align: center; margin: 30px 0;">
    <img src="./assets/img/architecture.svg" alt="TexDSP architecture" style="max-width: 100%; height: auto;">
  </div>
  <p style="text-align: center; font-size: 0.85em; color: #666;"><strong>Figure 1.1.</strong> <code>TexDSP</code> architecture diagram. </p>  
</div>

<div style="overflow-x: auto; max-width: 80%; margin: 0 auto; padding: 10px; box-sizing: border-box;">
  <div style="display: grid; grid-template-columns: repeat(3, minmax(200px, 1fr)); gap: 20px; text-align: center;">

  <!-- Header Row -->
  <div style="font-weight: bold;"><strong>Fire Model</strong></div>
  <div style="font-weight: bold;"><strong>Water Model</strong></div>
  <div style="font-weight: bold;"><strong>Wind Model</strong></div>

  <!-- Model Images Row -->
  <div><img src="./assets/img/fire.gif" alt="Fire Model" style="max-width: 100%;" /></div>
  <div><img src="./assets/img/bubbles_2.gif" alt="Bubbles Model" style="max-width: 100%;" /></div>
  <div><img src="./assets/img/wind.gif" alt="Wind Model" style="max-width: 100%;" /></div>

  <!-- First Sample Row -->
  <div>
    <audio controls style="width: 100%;">
      <source src="./assets/audios/texdsp_timbre_transfer/fire_to_fire.mp3" type="audio/mpeg" />
      Your browser does not support the audio element.
    </audio>
  </div>
  <div>
    <audio controls style="width: 100%;">
      <source src="./assets/audios/texdsp_timbre_transfer/bubbles_to_bubbles.mp3" type="audio/mpeg" />
      Your browser does not support the audio element.
    </audio>
  </div>
  <div>
    <audio controls style="width: 100%;">
      <source src="./assets/audios/texdsp_timbre_transfer/wind_to_wind.mp3" type="audio/mpeg" />
      Your browser does not support the audio element.
    </audio>
  </div>
</div>
</div>

<div style="overflow-x: auto; max-width: 80%; margin: 0 auto; padding: 10px; box-sizing: border-box; text-align: center; font-size: 0.85em; ">
<strong>Sound Examples 1.1.</strong> Sound examples generated using the <code>TexDSP</code> architecture trained using the <code>TexStat</code> loss function.</div>

