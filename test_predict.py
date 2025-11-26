import sys
from app import ECGPredictor
import numpy as np
import os

def make_panels_from_image(img_path, count=12):
    # return a list of length `count` containing the same image path
    return [img_path] * count


def summarize_wave(wave):
    # wave: (12, T)
    print('Wave shape:', wave.shape)
    for i in range(12):
        a = wave[i]
        print(f'Lead {i}: mean={a.mean():.4f} std={a.std():.4f} min={a.min():.4f} max={a.max():.4f} first10={a[:10].tolist()}')
    means = wave.mean(axis=1)
    print('Per-lead means:', means)
    print('Per-lead mean std:', means.std())


def main(img_paths):
    if not img_paths:
        print('Usage: python test_predict.py <image1.png> [image2.png ...]')
        sys.stdout.flush()
        return

    try:
        print('Loading model...')
        sys.stdout.flush()
        model = ECGPredictor('resnet_unet_best.pth')
        print('Model loaded successfully.')
        sys.stdout.flush()
    except Exception as e:
        print(f'Error loading model: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        return

    preds = []
    for p in img_paths:
        if not os.path.isfile(p):
            print(f"File not found: {p}")
            sys.stdout.flush()
            continue
        print('\n--- Predicting for', p)
        sys.stdout.flush()
        try:
            panels = make_panels_from_image(p, 12)
            wave = model.predict(panels)
            summarize_wave(wave)
            preds.append((p, wave))
        except Exception as e:
            print(f'Error predicting for {p}: {e}', file=sys.stderr)
            import traceback
            traceback.print_exc()
            continue
        sys.stdout.flush()

    # pairwise comparisons
    if len(preds) > 1:
        print('\nPairwise comparisons:')
        sys.stdout.flush()
        for i in range(len(preds)):
            for j in range(i+1, len(preds)):
                p1, w1 = preds[i]
                p2, w2 = preds[j]
                # align length
                T = min(w1.shape[1], w2.shape[1])
                w1s = w1[:, :T]
                w2s = w2[:, :T]
                # compute per-lead Pearson correlation (flattened)
                corrs = []
                dists = []
                for L in range(12):
                    a = w1s[L]
                    b = w2s[L]
                    if np.std(a) == 0 or np.std(b) == 0:
                        corr = 0.0
                    else:
                        corr = np.corrcoef(a, b)[0,1]
                    corrs.append(corr)
                    dists.append(np.linalg.norm(a - b))
                print(f"{os.path.basename(p1)} vs {os.path.basename(p2)} — mean corr: {np.nanmean(corrs):.4f}, std corr: {np.nanstd(corrs):.4f}, mean L2: {np.mean(dists):.4f}")
        sys.stdout.flush()
    else:
        print(f'Only {len(preds)} prediction(s) available; need at least 2 for pairwise comparison.')
        sys.stdout.flush()


if __name__ == '__main__':
    main(sys.argv[1:])
