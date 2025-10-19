#!/usr/bin/env python3
"""
auto_clip_ai_final.py — Versione completa BLIP + CLIP multimodale locale
con gestione prompt multilinea, selezione automatica e export FFmpeg.
"""
import os, sys, threading, cv2, tempfile, subprocess, json
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext, simpledialog, Toplevel, Button, Label
from PIL import Image, ImageTk
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
from sentence_transformers import util

# ----------------------
# CONFIG
# ----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_SKIP = 30
THUMB_SIZE = (320, 180)
SETTINGS_FILE = "settings.json"
ANALYSIS_FILE = "analysis.json"

# ----------------------
# FUNZIONE TOKENIZZAZIONE SICURA
# ----------------------
def safe_tokenize_lines(prompts):
    if prompts is None:
        raise ValueError("Il prompt non può essere None")
    if isinstance(prompts, str):
        lines = [line.strip() for line in prompts.split("\n") if line.strip()]
        if not lines:
            raise ValueError("Il prompt è vuoto dopo la pulizia")
    elif isinstance(prompts, list):
        lines = [line.strip() for line in prompts if isinstance(line, str) and line.strip()]
        if not lines:
            raise ValueError("La lista di prompt è vuota dopo la pulizia")
    else:
        raise TypeError(f"Tipo di input non valido: {type(prompts)}")
    return lines

# ----------------------
# AI MODELS
# ----------------------
class AIModels:
    def __init__(self):
        self.blip_model = None
        self.blip_processor = None
        self.clip_model = None
        self.clip_processor = None
        self.ready = False

    def load_models(self, progress_callback=None):
        if progress_callback: progress_callback("Caricamento BLIP... 0%")
        self.blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base", use_fast=True
        )
        self.blip_model.to(DEVICE)
        if progress_callback: progress_callback("BLIP pronto 50%")

        if progress_callback: progress_callback("Caricamento CLIP multimodale...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        if progress_callback: progress_callback("CLIP pronto 100%")
        self.ready = True

# ----------------------
# FUNZIONI AI
# ----------------------
def analyze_frame(frame, blip_model, blip_processor, clip_model, clip_processor, prompts, weight_blip=0.5):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)

    # --- BLIP ---
    inputs = blip_processor(images=pil_img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=30)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    blip_score = min(len(caption)/50.0,1.0)

    # --- CLIP multimodale ---
    clean_prompts = safe_tokenize_lines(prompts)
    inputs_clip = clip_processor(text=clean_prompts, images=pil_img, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = clip_model(**inputs_clip)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        clip_score = util.cos_sim(image_embeds, text_embeds).max().item()

    combined_score = weight_blip*blip_score + (1-weight_blip)*clip_score
    return caption, blip_score, clip_score, combined_score, image_embeds

def extract_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(str(video_path))
    frames=[]
    idx=0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if idx % FRAME_SKIP == 0:
            frames.append(frame.copy())
            if max_frames and len(frames)>=max_frames: break
        idx+=1
    cap.release()
    return frames
# ----------------------
# GUI
# ----------------------
class AutoClipGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Auto Clip AI Full Local CLIP")
        self.geometry("1600x950")

        self.ai_models = AIModels()
        self.videos=[]
        self.frames_dict={}
        self.scenes_dict={}
        self.tempdir=tempfile.mkdtemp()
        self.prompts=["paesaggio cinematografico","scena emozionante"]
        self.weight_blip=0.5

        self.create_widgets()
        self.load_models_async()
        self.load_settings()
        self.load_analysis()

    # GUI WIDGETS
    def create_widgets(self):
        top_frame = tk.Frame(self)
        top_frame.pack(fill="x", padx=5, pady=5)
        tk.Button(top_frame,text="➕ Aggiungi Video",command=self.add_videos,width=20,height=2).pack(side="left",padx=5)
        tk.Button(top_frame,text="Analizza Tutti",command=self.run_analysis,width=20,height=2).pack(side="left",padx=5)
        tk.Button(top_frame,text="Esporta Finale",command=self.export_final,width=20,height=2).pack(side="left",padx=5)
        tk.Button(top_frame,text="Gestione Prompt",command=self.edit_prompts,width=20,height=2).pack(side="left",padx=5)
        tk.Label(top_frame,text="Peso BLIP %:").pack(side="left", padx=(10,2))
        self.blip_slider=tk.Scale(top_frame,from_=0,to=100,orient="horizontal",length=150,command=self.update_weight)
        self.blip_slider.set(int(self.weight_blip*100))
        self.blip_slider.pack(side="left")

        self.terminal = scrolledtext.ScrolledText(self,height=14,font=("Courier",10))
        self.terminal.pack(fill="x",padx=5,pady=5)
        sys.stdout=self
        sys.stderr=self

        self.progress=ttk.Progressbar(self,orient="horizontal",length=800,mode="determinate")
        self.progress.pack(padx=5,pady=2)
        self.status_label=tk.Label(self,text="Caricamento modelli...")
        self.status_label.pack()

        # Canvas scorrevole
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.canvas = tk.Canvas(self.canvas_frame, bg="#f0f0f0")
        self.scrollbar = tk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0,0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.thumb_widgets = []

    # Terminal
    def write(self,msg):
        for line in str(msg).splitlines():
            if line.strip(): self.terminal.insert(tk.END,line+"\n")
        self.terminal.see(tk.END)
    def flush(self): pass

    # Caricamento modelli asincrono
    def load_models_async(self):
        def worker():
            self.ai_models.load_models(progress_callback=self.model_progress)
            self.status_label.config(text="Modelli pronti ✅")
        threading.Thread(target=worker,daemon=True).start()
    def model_progress(self,msg):
        self.status_label.config(text=msg)
        self.terminal.insert(tk.END,msg+"\n")
        self.terminal.see(tk.END)
        self.update_idletasks()

    # Prompt gestione
    def edit_prompts(self):
        win=Toplevel(self)
        win.title("Gestione Prompt CLIP")
        win.geometry("600x400")
        Label(win,text="Inserisci un prompt per riga:").pack(pady=5)
        text_area=scrolledtext.ScrolledText(win,width=70,height=20)
        text_area.pack(padx=10,pady=5,fill="both",expand=True)
        text_area.insert("1.0","\n".join(self.prompts))
        def save_and_close():
            new_text=text_area.get("1.0","end").strip()
            if new_text: self.prompts=[p.strip() for p in new_text.splitlines() if p.strip()]
            self.save_settings()
            win.destroy()
        Button(win,text="Salva",command=save_and_close).pack(pady=5)
    def update_weight(self,val):
        self.weight_blip=int(val)/100

    # Video
    def add_videos(self):
        files=filedialog.askopenfilenames(filetypes=[("Video files","*.mp4 *.mov *.avi")])
        for f in files:
            if f not in self.videos: self.videos.append(f)
        self.update_thumbs()

    def run_analysis(self):
        threading.Thread(target=self._analyze_thread,daemon=True).start()

    def _analyze_thread(self):
        total_videos=len(self.videos)
        for i,v in enumerate(self.videos,1):
            self.print_colored(f"[🔶 Analisi in corso] {os.path.basename(v)}","orange")
            frames=extract_frames(v)
            self.frames_dict[v]=frames
            self.scenes_dict[v]=[]
            last_embed=None
            for idx,f in enumerate(frames):
                caption, blip_s, clip_s, combined, embed = analyze_frame(
                    f,
                    self.ai_models.blip_model,
                    self.ai_models.blip_processor,
                    self.ai_models.clip_model,
                    self.ai_models.clip_processor,
                    self.prompts,
                    self.weight_blip
                )
                # Scarto frame simili (cos_sim > 0.95)
                selected=True
                if last_embed is not None:
                    if util.cos_sim(embed, last_embed).item()>0.95:
                        selected=False
                if selected: last_embed=embed

                self.scenes_dict[v].append((idx,caption,blip_s,clip_s,combined,selected))
                color="green" if selected else "black"
                self.print_colored(f"{os.path.basename(v):<30} | {idx:>4} | {combined:>4.2f} | {caption[:80]}",color)
                pct=int(((i-1+idx/len(frames))/total_videos)*100)
                self.progress["value"]=pct
                self.status_label.config(text=f"Analisi AI: {pct}%")
                self.update_idletasks()
            self.print_colored(f"[✅] {v} completato ({i}/{total_videos})","blue")
        self.update_thumbs()
        self.save_analysis()

    def print_colored(self,msg,color="black"):
        self.terminal.insert(tk.END,msg+"\n", (color,))
        self.terminal.tag_config(color,foreground=color)
        self.terminal.see(tk.END)

    # GUI Thumbnails
    def update_thumbs(self):
        for w in self.thumb_widgets: w.destroy()
        self.thumb_widgets.clear()
        for v in self.videos:
            hdr=tk.Label(self.scrollable_frame,text=os.path.basename(v),font=("Arial",12,"bold"))
            hdr.pack(fill="x",pady=(5,2))
            self.thumb_widgets.append(hdr)
            scenes=self.scenes_dict.get(v,[])
            for idx,caption,blip_s,clip_s,combined,selected in scenes:
                if selected: self.add_thumb_row(v,idx,caption,blip_s,clip_s,combined,selected)

    def add_thumb_row(self,video,idx,caption,blip_s,clip_s,combined,selected):
        row=tk.Frame(self.scrollable_frame,bd=1,relief="groove")
        try:
            frame=self.frames_dict[video][idx]
            img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img=Image.fromarray(img).resize(THUMB_SIZE)
            imgtk=ImageTk.PhotoImage(img)
        except Exception:
            imgtk=None
        lbl=tk.Label(row,image=imgtk)
        lbl.image=imgtk
        lbl.pack(side="left",padx=4,pady=4)
        info=tk.Frame(row)
        info.pack(side="left",padx=4)
        score_color="green" if selected else "black"
        tk.Label(info,text=f"Frame: {idx} | Score: {combined:.2f}",anchor="w",fg=score_color).pack(fill="x")
        tk.Label(info,text=f"{caption}",wraplength=400,justify="left",anchor="w").pack(fill="x")
        row.pack(fill="x",pady=2,padx=2)
        self.thumb_widgets.append(row)

    # Settings
    def save_settings(self):
        settings={"prompts":self.prompts,"weight_blip":self.weight_blip}
        with open(SETTINGS_FILE,"w") as f: json.dump(settings,f,indent=2)
    def load_settings(self):
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE,"r") as f:
                settings=json.load(f)
                self.prompts=settings.get("prompts",self.prompts)
                self.weight_blip=settings.get("weight_blip",self.weight_blip)
                self.blip_slider.set(int(self.weight_blip*100))
    def save_analysis(self):
        with open(ANALYSIS_FILE,"w") as f: json.dump(self.scenes_dict,f,indent=2)
    def load_analysis(self):
        if os.path.exists(ANALYSIS_FILE):
            with open(ANALYSIS_FILE,"r") as f:
                self.scenes_dict=json.load(f)

    # Export FFmpeg
    def export_final(self):
        outdir=filedialog.askdirectory(title="Cartella destinazione")
        if not outdir: return
        filename=simpledialog.askstring("Nome file","Inserisci il nome del file finale (senza estensione):",initialvalue="video_finale")
        if not filename: filename="video_finale"

        selected_clips=[]
        for v in self.videos:
            for idx,caption,blip_s,clip_s,combined,sel in self.scenes_dict.get(v,[]):
                if sel: selected_clips.append((v,idx))
        if not selected_clips:
            messagebox.showwarning("⚠ Nessun clip selezionato","Seleziona almeno un segmento da esportare.")
            return

        temp_clips=[]
        self.progress["value"]=0
        self.status_label.config(text="Esportazione FFmpeg: 0%")

        def ffmpeg_worker():
            total=len(selected_clips)
            for i,(v,idx) in enumerate(selected_clips,1):
                start=max((idx-1)*FRAME_SKIP/30,0)
                end=start+2*FRAME_SKIP/30
                temp_clip=os.path.join(self.tempdir,f"clip_{i:04d}.mp4")
                cmd=[
                    "ffmpeg","-y","-i",v,
                    "-ss",str(start),"-to",str(end),
                    "-vf","fps=30,scale=1280:-2",
                    "-c:v","libx264","-preset","fast","-crf","23",
                    "-c:a","aac","-b:a","128k",
                    temp_clip
                ]
                subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                temp_clips.append(temp_clip)
                pct=int(i/total*100)
                self.progress["value"]=pct
                self.status_label.config(text=f"Esportazione FFmpeg: {pct}%")
                self.update_idletasks()
            list_file=os.path.join(self.tempdir,"clips.txt")
            with open(list_file,"w") as f:
                for clip in temp_clips:
                    f.write(f"file '{clip}'\n")
            final_out=os.path.join(outdir,f"{filename}.mp4")
            subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",list_file,"-c","copy",final_out])
            self.progress["value"]=100
            self.status_label.config(text="Esportazione FFmpeg: 100%")
            messagebox.showinfo("✅ Esportazione completata",f"{final_out}")

        threading.Thread(target=ffmpeg_worker,daemon=True).start()

# ----------------------
# AVVIO
# ----------------------
if __name__=="__main__":
    app=AutoClipGUI()
    app.mainloop()

