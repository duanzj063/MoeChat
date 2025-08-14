document.addEventListener("DOMContentLoaded", () => {
  let mode = 'manual';
  let mediaRecorder;
  let audioChunks = [];
  let recording = false;
  let currentEventSource = null;
  let lastBotMessageDiv = null;
  const recordBtn = document.getElementById('recordBtn');
  const toggleModeBtn = document.getElementById('toggleModeBtn');
  const modeStatus = document.getElementById('modeStatus');
  const chatLog = document.getElementById('chatLog');
  const today = new Date(); // 愚人节彩蛋逻辑
  if (today.getMonth() === 3 && today.getDate() === 1) { // 4月1日
  const overlay = document.getElementById("crackOverlay");
  if (overlay) overlay.style.display = "block";
  }

  let audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  let isPlaying = false;
  const audioQueue = [];
  let isMuted = false;
  let activeGainNode = null;

  // ===== 新增：戳一戳功能的状态变量 =====
  let pokeCount = 0;
  let pokeResetTimer = null;
  // ===================================

  function enqueueAudio(base64String) {
    if (!base64String || base64String === "None" || base64String.length < 10) return;
    audioQueue.push(base64String);
    if (!isPlaying) playNextInQueue();
  }

  async function playNextInQueue() {
    if (audioQueue.length === 0) {
      isPlaying = false;
      return;
    }
    isPlaying = true;
    let base64String = audioQueue.shift();
    base64String = base64String.replace(/-/g, '+').replace(/_/g, '/');
    while (base64String.length % 4 !== 0) base64String += '=';
    try {
      const binaryString = atob(base64String);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) bytes[i] = binaryString.charCodeAt(i);
      const audioBuffer = await audioCtx.decodeAudioData(bytes.buffer);
      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      const gainNode = audioCtx.createGain();
      activeGainNode = gainNode;
      gainNode.gain.value = isMuted ? 0 : 1;
      source.connect(gainNode);
      gainNode.connect(audioCtx.destination);
      source.start(0);
      source.onended = () => playNextInQueue();
    } catch (err) {
      console.error("base64 解码失败:", err);
      playNextInQueue();
    }
  }

/*

███╗   ███╗ ██████╗ ███████╗ ██████╗██╗  ██╗ █████╗ ████████╗
████╗ ████║██╔═══██╗██╔════╝██╔════╝██║  ██║██╔══██╗╚══██╔══╝
██╔████╔██║██║   ██║█████╗  ██║     ███████║███████║   ██║   
██║╚██╔╝██║██║   ██║██╔══╝  ██║     ██╔══██║██╔══██║   ██║   
██║ ╚═╝ ██║╚██████╔╝███████╗╚██████╗██║  ██║██║  ██║   ██║   
╚═╝     ╚═╝ ╚═════╝ ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   

== MoeChat 前端核心控制文件 ==
== 为你打造情绪拉满、特效飞舞的聊天体验 ==
== ！==

📌 模块说明：
- 支持自动/手动录音识别 🎙️
- 特效触发：爱心💖、心碎💔、雨雪🌧️❄️、烟花🎆 等等


Tips:
  看见Bug别怕，它只是想被你 debug！

*/

//消息和头像
function appendMessage(role, text, append = false) {
  const now = new Date();
  const div = document.createElement("div");
  const timestamp = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
  div.className = `message ${role}`;
  const avatar = role === "user" ? "static/user__avatar.png" : "static/bot__avatar.png";
  const avatarHTML = `<img src="${avatar}" class="avatar ${role}-avatar">`;

  const imgTagRegex = /\{img\}((https?:\/\/[^\s]+)|\/[^\s]+)/;
  const imgMatch = text.match(imgTagRegex);

  if (role === "bot" && imgMatch) {
    const imgUrl = imgMatch[1]; 
    const remainingText = text.replace(imgTagRegex, "").trim();

    div.innerHTML = `${avatarHTML}<div class="bubble">
                        <small style='opacity: 0.6;'>[${timestamp}]</small><br>
                        <img src="${imgUrl}" alt="图片加载失败" style="max-width: 90%; border-radius: 8px; margin-top: 5px;">
                        ${remainingText ? `<br>${remainingText}` : ''}
                     </div>`;
  } else {
    if (role === "user") {
      div.innerHTML = `${avatarHTML}<div class="bubble"><small style='opacity: 0.6;'>[${timestamp}]</small><br>${text}</div>`;
    } else {
      div.innerHTML = `${avatarHTML}<div class="bubble"><small style='opacity: 0.6;'>[${timestamp}]</small><br>${text}</div>`;
    }
  }

  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;

  if (
  role === "user" &&
  text.includes("我") &&
  text.includes("爱") &&
  text.includes("你") &&
  !(
    text.includes("不") ||
    text.includes("没") ||
    text.includes("过")
  )
  ) {
  showHearts();
  }

  if (
    role === "user" &&
    (
      (text.includes("我") && text.includes("恨") && text.includes("你"))
      ||
      (text.includes("我") && text.includes("不") && text.includes("爱") && text.includes("你"))
      ||
      (text.includes("我") && text.includes("爱") && text.includes("过") && text.includes("你"))
    )
    ) {
    showBrokenHearts();
    }

  if (
  role === "user" &&
  (
    text.includes("早安") ||
    (text.includes("早") && text.includes("上") && text.includes("好"))
  )
  ) {
  showSunshine();
  }

if (
  ( (text.includes("生日") && text.includes("快乐")) || text.includes("生日愉快") )
  || text.includes("长命百岁")
  || text.includes("庆祝")
  || text.toLowerCase().includes("happy birthday")
  ) {
  launchBirthdayConfetti();
  }

  if (
  text.includes("下雨") ||
  text.toLowerCase().includes("rain")
  ) {
  launchRainEffect();
  }

  if (text.includes("下雪") || 
  text.toLowerCase().includes("snow")
  ) {
  launchSnowEffect();
  }

  if (text.includes("猜")) {
  launchFogEffect();
  }

  if (
  text.includes("晚安") ||
  text.toLowerCase().includes("good night") ||
  text.toLowerCase() === "gn"
  ) {
  launchGoodnightParticles();
  }

  if (
  text.includes("喜欢你") ||
  text.includes("暗恋你") ||
  text.includes("喜欢我")
  ) {
  launchRosePetals();
  }

  if ((/(哈|嘿|咯)/.test(text) || text.includes("😂"))) {
  const matchLaughs = text.match(/(哈|嘿|咯)/g) || [];
  const emojiCount = (text.match(/😂/g) || []).length;
  const totalLines = Math.min(1 + Math.floor((matchLaughs.length + emojiCount) / 2), 8);
  const direction = role === "user" ? "right" : "left";
  launchLaughEmojis(totalLines, direction);
  }

  if (text.includes("呜")) {
  const matchWuu = text.match(/呜/g) || [];
  const totalLines = Math.min(1 + Math.floor(matchWuu.length / 2), 8);
  const direction = role === "user" ? "right" : "left";
  launchCryEmojis(totalLines, direction);
  }

  const keywords = [
  "你妈的", "他妈的", "你他妈的", "他他妈的","我他妈","烦","死","妈逼",
  "你大爷的", "他大爷的", "傻逼", "白痴", "智障", "大爷的"
  ];
  const matchCount = keywords.reduce((acc, word) => acc + (text.includes(word) ? 1 : 0), 0);
    if (matchCount > 0) {
      const totalLines = Math.min(2 + matchCount * 2, 10);
      const direction = role === "user" ? "right" : "left";
    launchAngryEmojis(totalLines, direction);
    }

    if (text.includes("人家") || text.includes("害羞")) {
  const base = (text.includes("人家") ? 1 : 0) + (text.includes("害羞") ? 1 : 0);
  const totalLines = Math.min(1 + base * 3, 8);
  const direction = role === "user" ? "right" : "left";
  launchShyEmojis(totalLines, direction);
  }

  if (
  text.includes("春节") ||
  text.includes("烟花") ||
  text.includes("庆祝")
  ) {
  launchCustomFireworks();
  }

  if (role === "bot") lastBotMessageDiv = div.querySelector(".bubble");
  }

  let isHandling = false;
  function handleBlob(blob) {
    if (isHandling) return;
    isHandling = true;
    console.log("开始处理音频数据，blob大小:", blob.size);
    const reader = new FileReader();
    reader.onloadend = async () => {
      const base64AudioWithHeader = reader.result;
      console.log("音频数据转换为base64完成，开始发送到服务器");
      recordBtn.disabled = true;
      try {
        const res = await fetch('/web/audio', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ audio: base64AudioWithHeader })
        });
        const result = await res.json();
        console.log("服务器返回结果:", result);
        recordBtn.disabled = false;
        if (!result.text || result.text === 'null') {
          isHandling = false;
          return;
        }
        appendMessage("user", result.text);
        if (currentEventSource) currentEventSource.close();
        lastBotMessageDiv = null;
        currentEventSource = new EventSource('/web/stream_chat?text=' + encodeURIComponent(result.text));
        currentEventSource.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.done) {
            currentEventSource.close();
            currentEventSource = null;
            isHandling = false;
            return;
          }
          if (data.file && typeof data.file === 'string' && data.file.length > 20) {
            enqueueAudio(data.file);
          }
          if (data.message && typeof data.message === 'string') {
            appendMessage("bot", data.message, true);
          }
        };
        currentEventSource.onerror = (err) => {
          console.error('SSE error:', err);
          if (currentEventSource) currentEventSource.close();
          currentEventSource = null;
          isHandling = false;
        };
      } catch (error) {
        console.error("音频处理错误:", error);
        recordBtn.disabled = false;
        isHandling = false;
      }
    };
    reader.readAsDataURL(blob);
  }

  // === 录音按钮绑定 ===
  recordBtn.addEventListener('mousedown', () => {
    if (!recording) {
      audioChunks = [];
      mediaRecorder.start();
      recording = true;
      recordBtn.classList.add('recording');
    }
  });
  recordBtn.addEventListener('mouseup', () => {
    if (recording) {
      mediaRecorder.stop();
      recording = false;
      recordBtn.classList.remove('recording');
    }
  });
  recordBtn.addEventListener('touchstart', (e) => {
    e.preventDefault(); // 防止默认行为
    if (!recording) {
      audioChunks = [];
      mediaRecorder.start();
      recording = true;
      recordBtn.classList.add('recording');
    }
  });
  recordBtn.addEventListener('touchend', (e) => {
    e.preventDefault(); // 防止默认行为
    if (recording) {
      mediaRecorder.stop();
      recording = false;
      recordBtn.classList.remove('recording');
    }
  });

  navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    console.log("麦克风权限获取成功，音频流已建立");
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
    mediaRecorder.onstop = () => {
      console.log("录音停止，开始处理音频数据");
      handleBlob(new Blob(audioChunks, { type: 'audio/webm' }));
    };
    mediaRecorder.onerror = (e) => {
      console.error("MediaRecorder 错误:", e);
    };

    // === VAD 自动录音功能 ===
    let vadCtx, vadStream, vadSrc, vadAnalyser, vadData;
    let vadRecorder;
    let isSpeaking = false;
    let silenceStart = null;

    toggleModeBtn.onclick = () => {
      if (mode === 'manual') {
        mode = 'vad';
        modeStatus.textContent = '当前模式：自动识别';
        toggleModeBtn.textContent = '切换到手动录音模式';
        vadStream = stream;
        vadCtx = new AudioContext();
        vadSrc = vadCtx.createMediaStreamSource(stream);
        vadAnalyser = vadCtx.createAnalyser();
        vadAnalyser.fftSize = 512;
        vadData = new Uint8Array(vadAnalyser.fftSize);
        vadSrc.connect(vadAnalyser);
        vadRecorder = new MediaRecorder(stream);
        vadRecorder.ondataavailable = e => audioChunks.push(e.data);
        vadRecorder.onstop = () => handleBlob(new Blob(audioChunks, { type: 'audio/webm' }));

        function monitor() {
          vadAnalyser.getByteTimeDomainData(vadData);
          let sum = 0;
          for (let i = 0; i < vadData.length; i++) {
            const val = (vadData[i] - 128) / 128;
            sum += val * val;
          }
          const volume = Math.sqrt(sum / vadData.length);
          const now = Date.now();
          if (volume > 0.02 && !isSpeaking) {
            isSpeaking = true;
            silenceStart = null;
            audioChunks = [];
            vadRecorder.start();
            recordBtn.classList.add('recording');
          } else if (volume < 0.01 && isSpeaking && vadRecorder.state === 'recording') {
            if (!silenceStart) silenceStart = now;
            if (now - silenceStart > 500) {
              isSpeaking = false;
              vadRecorder.stop();
              recordBtn.classList.remove('recording');
            }
          } else {
            silenceStart = null;
          }
          if (mode === 'vad') requestAnimationFrame(monitor);
        }
        monitor();
      } else {
        mode = 'manual';
        modeStatus.textContent = '当前模式：手动录音';
        toggleModeBtn.textContent = '切换到自动识别模式';
        if (isSpeaking) {
          isSpeaking = false;
          vadRecorder.stop();
          recordBtn.classList.remove('recording');
        }
      }
    };
  });

  // ===== 新增：用于发送隐藏事件的辅助函数 =====
  function sendHiddenEvent(text) {
    if (!text || isHandling) return;
    isHandling = true;
    // 隐藏事件不在用户侧生成聊天气泡
    if (currentEventSource) currentEventSource.close();
    lastBotMessageDiv = null;
    currentEventSource = new EventSource('/web/stream_chat?text=' + encodeURIComponent(text));
    currentEventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.done) {
        currentEventSource.close();
        currentEventSource = null;
        isHandling = false;
        return;
      }
      if (data.file && typeof data.file === 'string' && data.file.length > 20) {
        enqueueAudio(data.file);
      }
      if (data.message && typeof data.message === 'string') {
        appendMessage("bot", data.message, true);
      }
    };
    currentEventSource.onerror = (err) => {
      console.error('SSE error:', err);
      if (currentEventSource) currentEventSource.close();
      currentEventSource = null;
      isHandling = false;
    };
  }
  // ===========================================

  document.getElementById('clearBtn').addEventListener('click', () => {
    chatLog.innerHTML = '';
    lastBotMessageDiv = null;
  });

  document.getElementById('muteToggle').addEventListener('click', () => {
    isMuted = !isMuted;
    document.getElementById('muteToggle').textContent = isMuted ? '🔈 开启播放' : '🔇 静音播放';
    if (activeGainNode) activeGainNode.gain.value = isMuted ? 0 : 1;
  });

  // ===== 新增：“戳一戳”的事件监听器 =====
  chatLog.addEventListener('click', (e) => {
    // 使用事件委托，判断点击的是否为机器人的头像
    if (e.target && e.target.matches('.avatar.bot-avatar')) {
      // 如果当前正在处理消息，则不响应点击
      if (isHandling) {
        console.log("正在处理消息，请稍后再戳。");
        return;
      }

      // 如果是新序列的第一次点击
      if (pokeCount === 0) {
        pokeCount = 1; // 开始计数
        // 设置5分钟的倒计时，到时后自动清零
        pokeResetTimer = setTimeout(() => {
          pokeCount = 0;
          console.log("5分钟无操作，戳一戳计数已清零。");
        }, 300000); // 5分钟 = 300,000毫秒

        console.log("5分钟内第1次，触发“害羞”事件。");
        sendHiddenEvent("{event:poke_shy}");

      } else { // 如果在5分钟窗口期内再次点击
        pokeCount++; // 累加计数
        
        if (pokeCount === 2) {
          console.log("5分钟内第2次，触发“疑问”事件。");
          sendHiddenEvent("{event:poke_confused}");
        } else { // 第3次及以后
          console.log(`5分钟内第${pokeCount}次，触发“生气”事件。`);
          sendHiddenEvent("{event:poke_angry}");
        }
      }
    }
  });
  // =======================================

  document.getElementById('manualSend').addEventListener('click', () => {
    const text = document.getElementById('manualInput').value.trim();
    if (!text || isHandling) return;
    isHandling = true;
    appendMessage("user", text);
    if (currentEventSource) currentEventSource.close();
    lastBotMessageDiv = null;
    currentEventSource = new EventSource('/web/stream_chat?text=' + encodeURIComponent(text));
    currentEventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.done) {
        currentEventSource.close();
        currentEventSource = null;
        isHandling = false;
        return;
      }
      if (data.file && typeof data.file === 'string' && data.file.length > 20) {
        enqueueAudio(data.file);
      }
      if (data.message && typeof data.message === 'string') {
        appendMessage("bot", data.message, true);
      }
    };
    currentEventSource.onerror = (err) => {
      console.error('SSE error:', err);
      if (currentEventSource) currentEventSource.close();
      currentEventSource = null;
      isHandling = false;
    };
    document.getElementById('manualInput').value = '';
  });

  // === 主题切换功能 ===
  const themeToggleBtn = document.getElementById('themeToggle');
  document.body.classList.add('light-mode');
  themeToggleBtn.addEventListener('click', () => {
    const isDark = document.body.classList.contains('dark-mode');
    document.body.classList.toggle('dark-mode', !isDark);
    document.body.classList.toggle('light-mode', isDark);
    themeToggleBtn.textContent = isDark ? '🌞 切换夜间模式' : '🌙 切换白天模式';
  });

//爱心粒子特效
  function showHearts(count = 32) { // count=密度（越大越密）
  let container = document.querySelector('.hearts-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'hearts-container';
    document.body.appendChild(container);
  }
  for (let i = 0; i < count; i++) {
    const heart = document.createElement('span');
    heart.className = 'heart-particle';
    heart.innerHTML = `<svg width="28" height="26" viewBox="0 0 32 29.6" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M23.6,0c-2.7,0-5.1,1.6-6.6,4C15.5,1.6,13.1,0,10.4,0C4.7,0,0,4.7,0,10.4c0,6.6,10.2,14.2,15.1,18
      c0.6,0.5,1.5,0.5,2.1,0c4.9-3.8,15.1-11.4,15.1-18C32,4.7,27.3,0,23.6,0z" fill="#ff4d6d" />
    </svg>`;
    heart.style.left = `${Math.random() * 92 + 1}vw`;
    heart.style.bottom = `${Math.random() * 16 + 2}vh`;
    heart.style.opacity = (0.75 + Math.random() * 0.22).toFixed(2);
    heart.style.transform = `scale(${0.8 + Math.random() * 0.7}) rotate(${Math.random() * 24 - 12}deg)`;
    heart.style.animationDelay = `${Math.random() * 0.7}s`;
    container.appendChild(heart);
    setTimeout(() => heart.remove(), 2400); // 2.4s动画
  }
}

//心碎粒子特效
function showBrokenHearts(count = 32) {
  let container = document.querySelector('.broken-hearts-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'broken-hearts-container';
    document.body.appendChild(container);
  }
  // 横向均匀分布
  for (let i = 0; i < count; i++) {
    const heart = document.createElement('span');
    heart.className = 'broken-heart-particle';
    heart.textContent = '💔';
    // 横坐标分层均匀，一点随机
    const baseLeft = (i + 0.5) * (100 / count); // 分布于0~100vw间
    const jitter = (Math.random() - 0.5) * (100 / count) * 0.6; // 少量左右抖动
    heart.style.left = `calc(${baseLeft}% + ${jitter}vw)`;
    heart.style.top = `-${Math.random() * 8 + 2}vh`;
    heart.style.fontSize = `${28 + Math.random()*20}px`;
    heart.style.animationDelay = `${Math.random() * 0.6}s`;
    container.appendChild(heart);
    setTimeout(() => heart.remove(), 2600);
  }
}
//早安
function showSunshine(count = 11) {
  let container = document.querySelector('.sunshine-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'sunshine-container';
    document.body.appendChild(container);
  }
  for (let i = 0; i < count; i++) {
    const sun = document.createElement('span');
    sun.className = 'sunshine';
    sun.textContent = '☀';
    // 弹跳延迟和大小透明度依然带随机
    sun.style.animationDelay = `${0.18 * i + Math.random() * 0.10}s`;
    sun.style.fontSize = `${34 + Math.random()*26}px`;
    sun.style.opacity = (0.78 + Math.random() * 0.2).toFixed(2);

    // 左右分布
    // 轨迹基于动画控制，这里可微调初始y
    sun.style.top = `${54 + (Math.random() - 0.5) * 8}vh`;
    sun.style.left = `-${18 + Math.random()*12}px`;
    container.appendChild(sun);
    setTimeout(() => sun.remove(), 4400); // 时长略大于动画
  }
  setTimeout(() => showCoffeeRow(), 3700);
}


function showCoffeeRow(count = 7) {
  let container = document.querySelector('.coffee-container');
  if (!container) {
    container = document.createElement('div');
    container.className = 'coffee-container';
    document.body.appendChild(container);
  }
  for (let i = 0; i < count; i++) {
    const cup = document.createElement('span');
    cup.className = 'coffee-cup';
    cup.textContent = '☕';
    // 均匀分布一排，横坐标由动画控制
    cup.style.bottom = '2px';
    cup.style.fontSize = `34px`; // 大小
    cup.style.left = `${8 + i*9 + Math.random()*2}vw`;
    cup.style.animationDelay = `${i * 0.14 + Math.random() * 0.06}s`;
    container.appendChild(cup);
    setTimeout(() => cup.remove(), 1700);
  }
}

//发射发射发射发射发射,默认1秒喷7次，每次左右各42颗
function launchBirthdayConfetti(burstCount = 7, interval = 150, particleCount = 42) {
  let times = 0;
  const spray = () => {
    // 左下角
    confetti({
      particleCount: particleCount,
      angle: 60,
      spread: 50,
      origin: { x: 0, y: 1 }
    });
    // 右下角
    confetti({
      particleCount: particleCount,
      angle: 120,
      spread: 50,
      origin: { x: 1, y: 1 }
    });
  };
  spray(); // 先喷一次
  times++;
  const timer = setInterval(() => {
    spray();
    times++;
    if (times >= burstCount) clearInterval(timer);
  }, interval);
}


// ===== 下雨特效=====
function launchRainEffect() {
  if (window.rainEffectLoaded) return;
  window.rainEffectLoaded = true;

  const rainScript = document.createElement("script");
  rainScript.src = "static/rain_effect.js";
  rainScript.id = "rainEffectScript";
  document.body.appendChild(rainScript);

  // 10 秒后自动清除 canvas 和标记
  setTimeout(() => {
    const canvas = document.querySelector("canvas");
    if (canvas && canvas.parentNode) canvas.parentNode.removeChild(canvas);
    const script = document.getElementById("rainEffectScript");
    if (script && script.parentNode) script.parentNode.removeChild(script);
    window.rainEffectLoaded = false;
  }, 10000); //播放十秒
}
// 下雪特效
function launchSnowEffect() {
  if (window.snowEffectLoaded) return;
  window.snowEffectLoaded = true;

  const snowContainer = document.createElement("div");
  snowContainer.id = "snow-container";
  snowContainer.style.position = "fixed";
  snowContainer.style.top = 0;
  snowContainer.style.left = 0;
  snowContainer.style.width = "100%";
  snowContainer.style.height = "100%";
  snowContainer.style.pointerEvents = "none";
  snowContainer.style.zIndex = 999;
  document.body.appendChild(snowContainer);

  const snowflakes = [];
  const maxFlakes = 80;
  for (let i = 0; i < maxFlakes; i++) {
    const flake = document.createElement("div");
    flake.className = "snowflake";
    flake.textContent = "❄";
    flake.style.fontSize = `${Math.random() * 8 + 14}px`;
    flake.style.left = `${Math.random() * 100}vw`;
    flake.style.top = `${Math.random() * -100}vh`;
    flake.style.opacity = (Math.random() * 0.5 + 0.4).toFixed(2);
    flake.style.animation = `fall ${5 + Math.random() * 5}s linear infinite`;
    snowContainer.appendChild(flake);
    snowflakes.push(flake);
  }

  // 10 秒后移除雪花
setTimeout(() => {
  // 移除雪花容器
  if (snowContainer.parentNode) snowContainer.parentNode.removeChild(snowContainer);
  window.snowEffectLoaded = false;

  // 显示冰霜遮罩
  const frost = document.getElementById("frostOverlay");
  if (frost) {
    frost.classList.add("active");

    // 冰霜持续 10 秒后淡出
    setTimeout(() => frost.classList.remove("active"), 10000);
  }

}, 10000); // 雪花持续 10 秒

}

// 雾气特效

function launchFogEffect() {
  if (window.fogEffectLoaded) return;
  window.fogEffectLoaded = true;

  const fogOverlay = document.createElement("div");
  fogOverlay.id = "fog-overlay";
  fogOverlay.className = "fog-overlay fog-animated";

  // 浓雾感
  for (let i = 0; i < 3; i++) {
    const layer = document.createElement("div");
    layer.className = `fog-layer layer-${i}`;
    layer.style.opacity = 0;
    fogOverlay.appendChild(layer);

    // 动画样式
    requestAnimationFrame(() => {
      layer.style.transition = "opacity 0.8s ease-out";
      layer.style.opacity = "0.9";
    });
  }

  document.body.appendChild(fogOverlay);

  setTimeout(() => {
    if (fogOverlay.parentNode) fogOverlay.parentNode.removeChild(fogOverlay);
    window.fogEffectLoaded = false;
  }, 10000);
}

// 晚安粒子特效
function launchGoodnightParticles() {
  if (window.goodnightEffectActive) return;
  window.goodnightEffectActive = true;

  const container = document.createElement("div");
  container.id = "goodnight-container";
  container.style.position = "fixed";
  container.style.top = 0;
  container.style.left = 0;
  container.style.width = "100%";
  container.style.height = "100%";
  container.style.pointerEvents = "none";
  container.style.zIndex = 999;
  document.body.appendChild(container);

  const chars = ["🌙", "⭐", "✨"];
  const count = Math.floor(Math.random() * 20) + 20; // 20~40个

  for (let i = 0; i < count; i++) {
    const el = document.createElement("span");
    el.className = "goodnight-particle";
    el.textContent = chars[Math.floor(Math.random() * chars.length)];

    const left = Math.random() * 100;
    const size = 20 + Math.random() * 20;
    const duration = 8 + Math.random() * 5; // 8-13s
    const delay = Math.random() * 2;

    el.style.left = `${left}vw`;
    el.style.fontSize = `${size}px`;
    el.style.animation = `goodnight-fall ${duration}s ease-in ${delay}s forwards, goodnight-blink ${1.5 + Math.random()}s infinite ease-in-out`;

    container.appendChild(el);
  }

  // 12秒播放
  setTimeout(() => {
    if (container.parentNode) container.parentNode.removeChild(container);
    window.goodnightEffectActive = false;
  }, 12000);
}


// 玫瑰花瓣喷射
function launchRosePetals(count = 144) {
  const container = document.createElement("div");
  container.style.position = "fixed";
  container.style.top = 0;
  container.style.left = 0;
  container.style.width = "100%";
  container.style.height = "100%";
  container.style.pointerEvents = "none";
  container.style.zIndex = 999;
  document.body.appendChild(container);

  const emojis = ["🌹", "💮", "🌺"];

  for (let i = 0; i < count; i++) {
    const el = document.createElement("span");
    el.className = "rose-petal";
    el.textContent = emojis[Math.floor(Math.random() * emojis.length)];

    const fromLeft = i % 2 === 0;
    const startX = fromLeft ? -5 : 105;
    const startY = 100 + Math.random() * 5;
    const translateX = fromLeft ? 50 + Math.random() * 30 : -50 - Math.random() * 30;
    const translateY = -80 - Math.random() * 30;
    const size = 20 + Math.random() * 14;

    el.style.left = `${startX}vw`;
    el.style.bottom = `${Math.random() * 6 + 2}vh`;  
    el.style.fontSize = `${size}px`;
    el.style.transform = 'translate(0, 0) scale(1) rotate(0deg)';


    const rotate = (Math.random() * 720 - 360).toFixed(2);
    requestAnimationFrame(() => {
      el.style.transform = `translate(${translateX}vw, ${translateY}vh) scale(0.7) rotate(${rotate}deg)`;
      el.style.opacity = '0';
    });

    container.appendChild(el);
  }

  setTimeout(() => {
    if (container.parentNode) container.remove();
  }, 3000);
}

function detectLaughEmojiTrigger(text) {
  const hahaCount = (text.match(/哈/g) || []).length;
  const emojiCount = (text.match(/😂/g) || []).length;
  const totalLines = Math.min(Math.floor((hahaCount + emojiCount) / 2), 8);
  if (totalLines > 0) launchLaughEmojis(totalLines);
}
//哈哈哈哈哈
    function launchLaughEmojis(lines = 4, direction = "right") {
      const container = document.createElement("div");
      container.style.position = "fixed";
      container.style.top = 0;
      container.style.left = 0;
      container.style.width = "100%";
      container.style.height = "100%";
      container.style.pointerEvents = "none";
      document.body.appendChild(container);

      const emojis = ["😂", "🤣"];

      for (let i = 0; i < lines; i++) {
        const countPerLine = Math.floor(Math.random() * 5) + 2; // 每行 2~6 个
        for (let j = 0; j < countPerLine; j++) {
          const emoji = document.createElement("div");
          emoji.className = `emoji-fly emoji-fly-${direction}`;
          emoji.textContent = emojis[Math.floor(Math.random() * emojis.length)];

          const top = 5 + (i * (90 / lines)) + Math.random() * 2; // 均匀分布 + 微抖动
          const horizontalOffset = j * 36 + Math.random() * 10;
          const opacity = (0.6 + Math.random() * 0.4).toFixed(2);

          emoji.style.top = `${top}vh`;
          emoji.style.opacity = opacity;
          if (direction === "right") {
            emoji.style.right = `-${horizontalOffset}px`;
          } else {
            emoji.style.left = `-${horizontalOffset}px`;
          }

          container.appendChild(emoji);
          setTimeout(() => emoji.remove(), 4000);
        }
      }

      setTimeout(() => {
        if (container.parentNode) container.remove();
      }, 4200);
    }
//飞哭，呜呜
    function launchCryEmojis(lines = 4, direction = "right") {
      const container = document.createElement("div");
      container.style.position = "fixed";
      container.style.top = 0;
      container.style.left = 0;
      container.style.width = "100%";
      container.style.height = "100%";
      container.style.pointerEvents = "none";
      document.body.appendChild(container);

      const emojis = ["😭", "🥲"];

      for (let i = 0; i < lines; i++) {
        const countPerLine = Math.floor(Math.random() * 5) + 2; // 每行 2~6 个
        for (let j = 0; j < countPerLine; j++) {
          const emoji = document.createElement("div");
          emoji.className = `emoji-fly emoji-fly-${direction}`;
          emoji.textContent = emojis[Math.floor(Math.random() * emojis.length)];

          const top = 5 + (i * (90 / lines)) + Math.random() * 2;
          const horizontalOffset = j * 36 + Math.random() * 10;
          const opacity = (0.6 + Math.random() * 0.4).toFixed(2);

          emoji.style.top = `${top}vh`;
          emoji.style.opacity = opacity;
          if (direction === "right") {
            emoji.style.right = `-${horizontalOffset}px`;
          } else {
            emoji.style.left = `-${horizontalOffset}px`;
          }

          container.appendChild(emoji);
          setTimeout(() => emoji.remove(), 4000);
        }
      }

      setTimeout(() => {
        if (container.parentNode) container.remove();
      }, 4200);
    }

//超过3个“呜”同时下雨，暂时有问题，已注释！
       /*function handleCryTriggers(text, role) {
      if (text.includes("呜")) {
        const matchWuu = text.match(/呜/g) || [];
        const totalLines = Math.min(1 + Math.floor(matchWuu.length / 2), 8);
        const direction = role === "user" ? "right" : "left";
        launchCryEmojis(totalLines, direction);

        if (/呜{3,}/.test(text)) {
          launchRainEffect();
        }
      }
    }*/
    //暴怒表情
    function launchAngryEmojis(lines = 4, direction = "right") {
      const container = document.createElement("div");
      container.style.position = "fixed";
      container.style.top = 0;
      container.style.left = 0;
      container.style.width = "100%";
      container.style.height = "100%";
      container.style.pointerEvents = "none";
      document.body.appendChild(container);

      const emojis = ["🤯", "😡", "🤬"];

      for (let i = 0; i < lines; i++) {
        const countPerLine = Math.floor(Math.random() * 4) + 5; // 每行 5~8 个
        for (let j = 0; j < countPerLine; j++) {
          const emoji = document.createElement("div");
          emoji.className = `emoji-fly emoji-fly-${direction} emoji-fast`;
          emoji.textContent = emojis[Math.floor(Math.random() * emojis.length)];

          const top = 5 + (i * (90 / lines)) + Math.random() * 2;
          const horizontalOffset = j * 30 + Math.random() * 10;
          const opacity = (0.7 + Math.random() * 0.3).toFixed(2);

          emoji.style.top = `${top}vh`;
          emoji.style.opacity = opacity;
          if (direction === "right") {
            emoji.style.right = `-${horizontalOffset}px`;
          } else {
            emoji.style.left = `-${horizontalOffset}px`;
          }

          container.appendChild(emoji);
          setTimeout(() => emoji.remove(), 2500);
        }
      }

      setTimeout(() => {
        if (container.parentNode) container.remove();
      }, 2800);
    }

    // 害羞表情动画
    function launchShyEmojis(lines = 4, direction = "right") {
      const container = document.createElement("div");
      container.style.position = "fixed";
      container.style.top = 0;
      container.style.left = 0;
      container.style.width = "100%";
      container.style.height = "100%";
      container.style.pointerEvents = "none";
      document.body.appendChild(container);

      const emojis = ["🥰", "🥺"];

      for (let i = 0; i < lines; i++) {
        const countPerLine = Math.floor(Math.random() * 5) + 2; // 每行 2~6 个
        for (let j = 0; j < countPerLine; j++) {
          const emoji = document.createElement("div");
          emoji.className = `emoji-fly emoji-fly-${direction}`;
          emoji.textContent = emojis[Math.floor(Math.random() * emojis.length)];

          const top = 5 + (i * (90 / lines)) + Math.random() * 2;
          const horizontalOffset = j * 36 + Math.random() * 10;
          const opacity = (0.6 + Math.random() * 0.4).toFixed(2);

          emoji.style.top = `${top}vh`;
          emoji.style.opacity = opacity;
          if (direction === "right") {
            emoji.style.right = `-${horizontalOffset}px`;
          } else {
            emoji.style.left = `-${horizontalOffset}px`;
          }

          container.appendChild(emoji);
          setTimeout(() => emoji.remove(), 4000);
        }
      }

      setTimeout(() => {
        if (container.parentNode) container.remove();
      }, 4200);
    }


// 判断当前农历时间

function isSpringFestivalLanternPeriod() {
  const now = new Date();
  const year = now.getFullYear();

  const festivalRanges = {
    2026: ["2026-02-05", "2026-02-22"],
    2027: ["2027-01-24", "2027-02-10"],
    2028: ["2028-01-14", "2028-01-31"],
    2029: ["2029-02-01", "2029-02-18"],
    2030: ["2030-01-21", "2030-02-07"]
  };

  if (!(year in festivalRanges)) return false;

  const [startStr, endStr] = festivalRanges[year];
  const start = new Date(startStr);
  const end = new Date(endStr);

  return now >= start && now <= end;
}

// 显示遮罩（局部函数）
function showLanternOverlay() {
  const overlay = document.getElementById("lanternOverlay");
  if (overlay) overlay.style.display = "block";
}

// 页面加载后自动判断
document.addEventListener("DOMContentLoaded", () => {
  if (isSpringFestivalLanternPeriod()) {
    showLanternOverlay();
  }
});

// ===== 万圣节逻辑 =====

  function isHalloweenPeriod() {
    const now = new Date();
    const month = now.getMonth(); 
    const day = now.getDate();    
    return (month === 9 && (day === 30 || day === 31)) || (month === 10 && day === 1);
  } //10月30，31和11月01

  function showHalloweenOverlay() {
    const overlay = document.getElementById("halloweenOverlay");
    if (overlay) overlay.style.display = "block";
  }

  if (isHalloweenPeriod()) {
    showHalloweenOverlay();
  }

  // === 圣诞节遮罩逻辑 ===
  function isChristmasPeriod() {
    const now = new Date();
    const month = now.getMonth(); // 0-11
    const day = now.getDate();
    return (month === 11 && day >= 23 && day <= 27); // 12月23-27日
  }

  function showChristmasOverlay() {
    const overlay = document.getElementById("christmasOverlay");
    if (overlay) overlay.style.display = "block";
  }

  if (isChristmasPeriod()) {
    showChristmasOverlay();
  }

//烟花
  let fireworksInstance = null;

  function initFireworksCanvas() {
  if (document.getElementById("fireworksCanvas")) return;

  const canvas = document.createElement("canvas");
  canvas.id = "fireworksCanvas";
  canvas.style.position = "fixed";
  canvas.style.top = 0;
  canvas.style.left = 0;
  canvas.style.width = "100vw";
  canvas.style.height = "100vh";
  canvas.style.pointerEvents = "none";
  canvas.style.zIndex = 9999;
  document.body.appendChild(canvas);

  // 获取Fireworks构造函数？？？
  const FireworksConstructor = window.Fireworks?.default || window.Fireworks;
  if (!FireworksConstructor) {
    console.error("Fireworks 构造函数未找到！");
    return;
  }

  fireworksInstance = new FireworksConstructor(canvas, {
    autoresize: true,
    opacity: 0.1,
    acceleration: 1.05,
    friction: 0.97,
    gravity: 1.5,
    particles: 120,
    traceSpeed: 12,
    traceLength: 4,
    explosion: 6,
    brightness: { min: 60, max: 90 },
    decay: { min: 0.015, max: 0.03 },
    hue: { min: 0, max: 360 },
    flickering: 60,
    intensity: 25,
    lineStyle: "round",
    mouse: { click: false, move: false, max: 1 },
    delay: { min: 5, max: 15 },
    lineWidth: {
      trace: { min: 1, max: 2 },
      explosion: { min: 2, max: 3.5 },
    },
    boundaries: {
      x: 0,
      y: 20,
      width: window.innerWidth,
      height: window.innerHeight / 2,
    }
  });

  fireworksInstance.start();
}
//数量，引爆时机
  function launchCustomFireworks() {
    initFireworksCanvas();
    const count = Math.floor(Math.random() * 6) + 3; 
    for (let i = 0; i < count; i++) {
      setTimeout(() => {
        fireworksInstance.launch(1);
      }, Math.random() * 1000); // 1秒内随机引爆
    }
  }
});
