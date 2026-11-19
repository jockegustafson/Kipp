const $ = () => ({});
const FILLERS = new Set(["eh","ehm","äh","ähm","öh","öhm","mm","mmm","mhm","hm","hmm"]);
const CUES = new Set(["liksom","typ","alltså","asså","ba","ju","väl","nej","ja","okej","jaha","oj",
  "wallah","walla","yani","len","bre","mannen","aboow","khalas","ayde","jalla","yalla","tamam","shoo","salam","inshallah"]);
const SLANG = new Set(["abaayo", "abo", "aboow", "abow", "achi", "adey", "ahki", "aight", "aina", "ainasis", "ait", "ajaib", "ajde", "ajjayb", "ajt", "akhi", "alehom", "aleykum", "aläsas", "amo", "amu", "araba", "arre", "askar", "ayde", "azizam", "baelesh", "bajta", "bajtade", "bajtar", "bajtat", "bakk", "bast", "baxa", "baxxa", "baxxade", "baxxar", "baxxat", "bazza", "bazzade", "bazzar", "bazzat", "beckna", "becknade", "becknar", "becknat", "belesh", "beshlik", "billahi", "brakkam", "bram", "brat", "bre", "brusha", "brushan", "bull", "bäckna", "bäcknade", "bäcknar", "bäcknat", "bäzza", "bäzzade", "bäzzar", "bäzzat", "caadi", "cadaan", "chagga", "chaggaz", "chalas", "chappa", "chappade", "chappar", "chappat", "chilla", "chillade", "chillar", "chillat", "chisko", "chuno", "cok", "dadaash", "dadash", "damma", "dammade", "dammar", "dammat", "dein", "diin", "fi", "floss", "flous", "floz", "garac", "gendish", "git", "gitt", "gitta", "gittade", "gittar", "gittat", "gittish", "goare", "gola", "golade", "golar", "golat", "gus", "guss", "guz", "guzz", "gäri", "gäris", "göt", "götten", "güzz", "habibi", "haffa", "haffade", "haffar", "haffat", "haffish", "haffla", "hafflade", "hafflar", "hafflat", "hafla", "haflade", "haflar", "haflat", "haiwan", "hajde", "hakarat", "harakat", "harakatt", "hassla", "hasslade", "hasslar", "hasslat", "hawey", "hayat", "hayati", "haywan", "hazzla", "hazzlade", "hazzlar", "hazzlat", "hermano", "inshallah", "jalla", "jappa", "jappade", "jappar", "jappat", "jara", "jende", "jäni", "kanis", "kaoz", "katt", "keff", "kelb", "khalas", "khanis", "khoya", "kirra", "kische", "kishe", "klivare", "knas", "knatch", "köpek", "lacag", "lalla", "lallade", "lallar", "lallat", "lallish", "lattjo", "launa", "lax", "len", "livs", "livset", "loco", "majmun", "mariam", "masari", "mashakil", "mashallah", "mashara", "massari", "merci", "meskin", "miskin", "moré", "nacas", "nada", "olum", "opp", "ortenbarn", "para", "paraflow", "paraknas", "parra", "popo", "pröjsa", "saaxib", "salam", "saxib", "seldi", "shabab", "shababs", "shekorina", "shikorina", "shishko", "sho", "sholokh", "shonne", "shono", "shonu", "shoo", "shorta", "shouno", "shuna", "shunne", "shuno", "shunos", "shurda", "snea", "sneade", "snear", "sneat", "softa", "stasha", "stashade", "stashar", "stashat", "suedi", "svenne", "swedi", "tabanja", "tabbe", "tagga", "taggade", "taggar", "taggat", "tamam", "ti", "tjabo", "tjalla", "tjallade", "tjallar", "tjallat", "tjej", "vischan", "wacka", "wackade", "wackar", "wackat", "walla", "wallaahi", "wallah", "wallahi", "warya", "waryaa", "woria", "woriah", "yalla", "yani", "çok"]);   // lexicon_slang.dict (ortenslang) + slang words Braxen already had
const PHRASE_START = new Set(["och","men","att","som","när","om","för","eftersom","så","eller","fast","då","innan","medan","tills","därför","utan","vilket","där","dit"]);


function bare(w) { return w.toLowerCase().replace(/[^a-zåäöéü0-9]/g, ""); }
function syllables(w) {
  const b = bare(w); if (!b) return 0;
  if (/^\d+$/.test(b)) return 2 * b.length;
  const m = b.match(/[aeiouyåäöéü]+/g); return Math.max(1, m ? m.length : 1);
}
function voicelessShare(w) {
  const b = bare(w).replace(/sj|skj|stj|tj|ch|sch/g, "S"); if (!b) return 0;
  const n = (b.match(/[ptkfshcxS]/g) || []).length; return n / b.length;
}
function punctOf(w) { const m = w.match(/([.,!?;:–-])["')\]]*$/); return m ? m[1] : ""; }

// ---- the planner: lung volume over the words, a breath where the next stretch would go below the lowest volume ----------
function params() {
  return { vc: +$("vc").value, start: +$("start").value / 100, rel: +$("rel").value / 100, top: +$("top").value / 100,
           flow: +$("flow").value, eff: +$("eff").value, vl: +$("vl").value, rate: +$("rate").value,
           plan: $("plan").checked, first: $("first").checked, silentOk: $("silentOk").checked, place };
}
const PHON_FLOW = 0.11;   // L per second of voiced phonation at normal effort (the lung model's air unit)
function wordAir(dur, vl, P) { return dur * PHON_FLOW * P.eff * ((1 - vl) + vl * P.vl); }
function pauseAfter(p) { return p === "," ? 0.18 : p === "." || p === "!" || p === "?" ? 0.38 : p ? 0.22 : 0; }
function candidate(words, i, P) {   // may a breath come before word i?
  if (i === 0) return "start";
  const p = punctOf(words[i - 1].w);
  if (/[.!?]/.test(p)) return "sentence";
  if (p) return "punct";
  if (P.place === "punct") return null;
  const b = bare(words[i].w);
  if (P.place === "any") return "any";
  return PHRASE_START.has(b) || FILLERS.has(b) || CUES.has(b) ? "phrase" : null;
}
function classify(dv, dur, P, kind) {
  if (P.silentOk && dv < 0.22 && (kind === "punct" || kind === "phrase")) return "silent";
  const size = dur < 0.30 ? "S" : dur < 0.55 ? "M" : "L";
  const loud = P.eff >= 1.25 || P.flow >= 1.8 || dv / P.vc > 0.16;
  return size + (loud ? "loud" : "soft");
}
const EFF_MARK = { "låg": "low", "mellan": "mid", "hög": "high", low: "low", mid: "mid", high: "high" };
function effWords(text) {   // [låg] / [mellan] / [hög] set the effort of the words after them (2026-11-13)
  const out = []; let cur = null, changed = false;
  for (const w of text.split(/\s+/).filter(Boolean)) {
    const m = w.match(/^\[(\S+)\]$/), e = m && EFF_MARK[m[1].toLowerCase()];
    if (e) { changed = changed || e !== cur; cur = e; continue; }
    out.push({ w, eff: cur, effChange: changed && out.length > 0 }); changed = false;
  }
  return out;
}
function plan(text, P) {
  const words = effWords(text).map(({ w, eff, effChange }) => {
    const syl = syllables(w), vl = voicelessShare(w);
    return { w, eff, effChange, syl, vl, dur: syl ? syl / P.rate : 0.05, filler: FILLERS.has(bare(w)), cue: CUES.has(bare(w)), slang: SLANG.has(bare(w)) };
  });
  words.forEach(o => { if (o.filler) o.dur = Math.max(o.dur, 0.35); o.air = wordAir(o.dur, o.vl, P); });
  const VC = P.vc, low = P.rel * VC, top = P.top * VC;
  const cands = words.map((o, i) => candidate(words, i, P) || (o.effChange ? "sentence" : null));   // a breath at an effort change
  const airTo = (i, stopAt) => { let a = 0; for (let j = i; j < words.length; j++) { if (j > i && stopAt(j)) break; a += words[j].air; } return a; };
  let V = P.start * VC;
  for (let i = 0; i < words.length; i++) {
    const kind = cands[i];
    if (kind) {
      const toNext = airTo(i, j => !!cands[j]);
      const toSentence = airTo(i, j => cands[j] === "sentence" || cands[j] === "start");
      let breathe = false;
      if (kind === "start") breathe = P.first;
      else if (words[i].effChange) breathe = true;
      else if (V - toNext < low) breathe = true;                                    // would go below the lowest volume
      else if (kind === "sentence" && V - toSentence < low + 0.25 * (top - low)) breathe = true;   // a natural sentence breath
      if (breathe) {
        let target = top;
        if (P.plan) {    // size the breath to what follows: up to where the next breath is likely, plus a margin
          let a = 0;
          for (let j = i; j < words.length; j++) {
            if (j > i && cands[j] === "sentence" && a > 0.35 * (top - low)) break;
            if (a + words[j].air > top - low) break;
            a += words[j].air;
          }
          target = Math.min(top, Math.max(V + 0.12, low + a + 0.08 * VC));
        }
        const dv = Math.max(0, target - V);
        if (dv >= 0.05 || kind === "start" || words[i].effChange) {
          const dur = Math.max(0.15, dv / P.flow);
          words[i].br = classify(dv, dur, P, kind); words[i].dv = dv;
          words[i].bdur = words[i].br === "silent" ? Math.max(0.12, dur * 0.6) : dur;
          V = target;
        }
      }
    }
    V -= words[i].air;                                                                  // every word spends its air
  }
  return words;
}
// re-walk the planned breaths at the actual (or estimated) times -> volume curve points and times
function simulate(words, P, times) {
  const VC = P.vc; let V = P.start * VC, t = 0; const pts = [[0, V]], out = { words: [], breaths: [] };
  words.forEach((o, i) => {
    const wt = times && times.words[i], bt = times && times.breaths[i];
    if (o.br) {
      const s = bt ? bt.s : t, e = bt ? bt.e : t + o.bdur;
      pts.push([s, V]); V = Math.min(VC, V + o.dv); pts.push([e, V]); out.breaths[i] = { s, e, cls: o.br, dv: o.dv }; t = e;
    }
    const s = wt ? wt.s : t, e = wt ? wt.e : t + o.dur;   // not generated yet: continues from the last real time
    const air = times && wt ? wordAir(e - s, o.vl, P) : o.air;
    pts.push([s, V]); V -= air; pts.push([e, V]); out.words[i] = { s, e };
    // estimated pause after the word; shorter when a breath follows (the breath takes the pause)
    t = e + (wt ? 0 : pauseAfter(punctOf(o.w)) * (words[i + 1] && words[i + 1].br ? 0.3 : 1));
  });
  out.pts = pts; out.end = Math.max(t, ...pts.map(p => p[0])); out.minV = Math.min(...pts.map(p => p[1])); return out;
}


module.exports = { plan, simulate };
