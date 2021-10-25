import os
import pickle
import sys
import time
from subprocess import Popen, PIPE

import numpy as np
import torch
import torchvision.transforms.functional as TF
#from IPython.display import display
from PIL import Image
from torchvision.transforms import Compose, Resize
from tqdm import tqdm
import logging 
import utils 

logging.basicConfig(filename='./data/stylegan.log',
                            filemode='w',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)

class StyleGan():
    def __init__(self, ):

        self.logger = logging.getLogger('StyleGan3')
        self.load_env()
        self.device = self.load_device()
        self.tf = Compose([Resize(224), lambda x: torch.clamp((x + 1) / 2, min=0, max=1), ])
        self.download_models()
        self.timestring = None
        self.clip = utils.CLIP()

    def load_env(self):
        sys.path.append('./CLIP')
        sys.path.append('./stylegan3')
        self.logger.info('Added CLIP and Stylegan3 to path.')

    def load_device(self):
        device = torch.device('cuda:0')
        self.logger.info(f'Using device: {device}')
        return device

    def download_models(self):
        base_url = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/"
        model_name = {
            "FFHQ": "stylegan3-t-ffhqu-1024x1024.pkl",
            "MetFaces": "stylegan3-r-metfacesu-1024x1024.pkl",
            "AFHQv2": "stylegan3-t-afhqv2-512x512.pkl"
        }
        for model in model_name.items():
            if model_name[model[0]] not in os.listdir('./models'):
                network_url = base_url + model[1]
                utils.fetch_model(network_url)
                
                #with open(utils.fetch_model(network_url), 'rb') as fp:
                #    G = pickle.load(fp)['G_ema'].to(self.device)
                #with open("./data/models/"+str(model[0]),"wb") as handle:
                #    pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)
                self.logger.info(f'Model {model[0]} correctly downloaded.')
            else:
                self.logger.info(f'Model {model[0]} were previously downloaded.')

    def choose_model(self,model):  
        model_name = {
            "FFHQ": "stylegan3-t-ffhqu-1024x1024.pkl",
            "MetFaces": "stylegan3-r-metfacesu-1024x1024.pkl",
            "AFHQv2": "stylegan3-t-afhqv2-512x512.pkl"
        }
        self.logger.info(model+ model_name[model]+str(os.listdir("./models/"))+"./models/"+model_name[model])
        try:
            with open("./models/"+model_name[model], 'rb') as fp:
                G = pickle.load(fp)['G_ema'].to(self.device)
        except Exception as loadError:
            self.logger.error("Error loading the model"+str(loadError))
            return None, None

        zs = torch.randn([10000, G.mapping.z_dim], device=self.device)
        w_stds = G.mapping(zs, None).std(0)
        return G, w_stds

    def run(self, timestring):

        torch.manual_seed(self.seed)
        with torch.no_grad():
            qs = []
            losses = []
            for _ in range(8):
                q = (self.G.mapping(torch.randn([4, self.G.mapping.z_dim], device=self.device), None,
                                    truncation_psi=0.7) - self.G.mapping.w_avg) / self.w_stds
                images = self.G.synthesis(q * self.w_stds + self.G.mapping.w_avg)
                embeds = utils.embed_image(images.add(1).div(2))
                loss = utils.spherical_dist_loss(embeds, self.target).mean(0)
                i = torch.argmin(loss)
                qs.append(q[i])
                losses.append(loss[i])
            qs = torch.stack(qs)
            losses = torch.stack(losses)
            print(losses)
            print(losses.shape, qs.shape)
            i = torch.argmin(losses)
            q = qs[i].unsqueeze(0).requires_grad_()

            # Sampling loop
        q_ema = q
        opt = torch.optim.AdamW([q], lr=0.03, betas=(0.0, 0.999))
        loop = tqdm(range(self.steps))
        for i in loop:
            opt.zero_grad()
            w = q * self.w_stds
            image = self.G.synthesis(w + self.G.mapping.w_avg, noise_mode='const')
            embed = utils.embed_image(image.add(1).div(2))
            loss = utils.spherical_dist_loss(embed, self.target).mean()
            loss.backward()
            opt.step()
            loop.set_postfix(loss=loss.item(), q_magnitude=q.std().item())

            q_ema = q_ema * 0.9 + q * 0.1
            image = self.G.synthesis(q_ema * self.w_stds + self.G.mapping.w_avg, noise_mode='const')

            if i % 10 == 0:
                #display(TF.to_pil_image(self.tf(image)[0]))
                print(f"Image {i}/{self.steps} | Current loss: {loss}")
            pil_image = TF.to_pil_image(image[0].add(1).div(2).clamp(0, 1))
            os.makedirs(f'{self.path_2_save}/{timestring}', exist_ok=True)
            pil_image.save(f'{self.path_2_save}/{timestring}/{i:04}.jpg')

    def generate_video(self, timestring, video_length=15):
        frames = os.listdir(f"{self.path_2_save}/{timestring}")
        frames = len(list(filter(lambda filename: filename.endswith(".jpg"), frames)))

        init_frame = 1
        last_frame = frames

        min_fps = 10
        max_fps = 30

        total_frames = last_frame - init_frame

        frames = []
        tqdm.write('Generating video...')
        for i in range(init_frame, last_frame):
            filename = f"{self.path_2_save}/{timestring}/{i:04}.jpg"
            frames.append(Image.open(filename))

        fps = np.clip(total_frames / video_length, min_fps, max_fps)

        p = Popen(
            ['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(fps), '-i', '-', '-vcodec', 'libx264',
             '-r', str(fps), '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'veryslow', 'video.mp4'], stdin=PIPE)
        for im in tqdm(frames):
            im.save(p.stdin, 'PNG')
        p.stdin.close()

        print("The video is now being compressed, wait...")
        p.wait()
        print("The video is ready")

    def run_update(self, model, text, steps=200, seed=126, path_2_save="./data/"):

        self.text = text
        self.steps = steps
        self.seed = seed
        self.path_2_save = path_2_save
        self.G, self.w_stds = self.choose_model(model)
        self.target = self.clip.embed_text(text)

        self.timestring = time.strftime('%Y%m%d%H%M%S')
        self.run(self.timestring)
        self.generate_video(self.timestring)


if __name__ == "__main__":
    stylegan = StyleGan(model="FFHQ",
                        text="Monna Lisa portrait",
                        steps=200,
                        seed=126)
