"""
Dataset loader for the Volleyball dataset. 
non temporal (seq = False) return person_crops: (N, C, H, W) tensor of person crops for the frame
temporal (seq = True) return person_crops: (N, 9, C, H, W) tensor of person crops for the 9-frame clip centered around the frame_id
"""
import torch 
from torch.utils.data import Dataset
from pathlib import Path

import pickle
import cv2


PERSON_ACTIVITY_CLASSES = ["Waiting", "Setting", "Digging", "Falling", "Spiking", "Blocking", "Jumping", "Moving", "Standing"]
GROUP_ACTIVITY_CLASSES = ["r_set", "r_spike", "r-pass", "r_winpoint", "l_winpoint", "l-pass", "l-spike", "l_set"]

ACTIVITIES_LABELS = {
    "person": {name.lower(): i for i, name in enumerate(PERSON_ACTIVITY_CLASSES)},
    "group":  {name: i for i, name in enumerate(GROUP_ACTIVITY_CLASSES)}
}

class GroupActivityDataset(Dataset):
    def __init__(self, data_dir, annot_dir, split, seq=False, transform=None, return_person_labels=False):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.annot_dir = Path(annot_dir)
        self.split = split
        self.seq = seq
        self.transform = transform 
        self.return_person_labels = return_person_labels

        self.group_label_map = ACTIVITIES_LABELS["group"]
        self.person_label_map = ACTIVITIES_LABELS["person"]
        
        with open(self.annot_dir, 'rb') as f:
            self.videos_annot = pickle.load(f)

        self.data_samples = self._prepare_data()

    def _prepare_data(self):
        samples = []
        for video_id in self.split:
            for clip_dir, clip_info in self.videos_annot[str(video_id)].items():
                category = clip_info['category']

                for frame_id, boxes in clip_info['frame_boxes_dct'].items():
                    samples.append({
                        'video_id': video_id,
                        'clip_dir': clip_dir,
                        'frame_id': frame_id,
                        'boxes': boxes,
                        'group_category': category,
                    })
        return samples
    
    def _load_frame(self, video_id, clip_dir, frame_id):
        path = self.data_dir / str(video_id) / str(clip_dir) / f"{frame_id}.jpg"
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Failed to load frame: {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _crop_and_transform(self, frame, box):
        """Extract a person crop from the frame and apply the transform."""
        H_frame, W_frame = frame.shape[:2]
        x1, y1, x2, y2 = map(int, box.box)

        # Clamp coordinates to valid frame bounds
        x1 = max(0, min(x1, W_frame - 1))
        x2 = max(x1 + 1, min(x2, W_frame))
        y1 = max(0, min(y1, H_frame - 1))
        y2 = max(y1 + 1, min(y2, H_frame))

        crop = frame[y1:y2, x1:x2]
        if self.transform:
            crop = self.transform(image=crop)['image']
            if not isinstance(crop, torch.Tensor):
                crop = torch.from_numpy(crop).permute(2, 0, 1).float()
        else:
            crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
            crop = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        return crop

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]

        N = 12
        C, H, W = 3, 224, 224
        person_labels = torch.full((N,), -100, dtype=torch.long)

        sorted_boxes = sorted(sample['boxes'], key=lambda b: (b.box[0] + b.box[2]) / 2)

        if self.seq:
            person_crops = torch.zeros(N, 9, C, H, W)
            
            # Handle frame_id string and padding safely
            fid = sample['frame_id']
            fid_str = str(fid)
            pad_len = len(fid_str) if isinstance(fid, str) and fid_str.startswith('0') else 0
            base_fid = int(fid)
            
            for clip_idx in range(9):
                curr_fid = base_fid + clip_idx
                curr_fid_str = f"{curr_fid:0{pad_len}d}" if pad_len > 0 else str(curr_fid)
                
                clip_frame = self._load_frame(sample['video_id'], sample['clip_dir'], curr_fid_str)
                for slot_idx, box in enumerate(sorted_boxes):
                    if slot_idx >= N:
                        break
                    person_crops[slot_idx, clip_idx] = self._crop_and_transform(clip_frame, box)
        else:
            person_crops = torch.zeros(N, C, H, W)
            frame = self._load_frame(sample['video_id'], sample['clip_dir'], sample['frame_id'])
            for slot_idx, box in enumerate(sorted_boxes):
                if slot_idx >= N:
                    break
                person_crops[slot_idx] = self._crop_and_transform(frame, box)

        for slot_idx, box in enumerate(sorted_boxes):
            if slot_idx >= N:
                break
            category_key = box.category.lower()
            person_labels[slot_idx] = self.person_label_map[category_key]

        group_label = torch.tensor(self.group_label_map[sample['group_category']], dtype=torch.long)

        if self.return_person_labels:
            return person_crops, group_label, person_labels
        return person_crops, group_label


def collate_fn(batch):
    if len(batch) == 0:
        raise ValueError("Batch is empty")

    first_item_len = len(batch[0])

    if first_item_len == 2:
        person_crops_batch, group_labels_batch = zip(*batch)
        person_crops_batch = torch.stack(person_crops_batch) # shape: (batch_size, N, C, H, W)
        group_labels_batch = torch.stack(group_labels_batch) # shape: (batch_size,)
        return person_crops_batch, group_labels_batch

    if first_item_len == 3:
        person_crops_batch, group_labels_batch, person_labels_batch = zip(*batch)
        person_crops_batch = torch.stack(person_crops_batch) # shape: (batch_size, N, C, H, W)
        group_labels_batch = torch.stack(group_labels_batch) # shape: (batch_size,)
        person_labels_batch = torch.stack(person_labels_batch) # shape: (batch_size, N)
        return person_crops_batch, group_labels_batch, person_labels_batch

    raise ValueError(f"Unexpected batch item size: {first_item_len}")

