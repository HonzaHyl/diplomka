import os

directory = "/srv/home/jhyl/Afib_recurrence/finetune_data_all"

def flip_labels(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        changed = False
        for i, line in enumerate(lines):
            if line.startswith('#Dx:'):
                parts = line.split(': ')
                if len(parts) > 1:
                    labels = parts[1].split(',')
                    new_labels = []
                    for label in labels:
                        clean_label = label.strip()
                        if clean_label == '0':
                            new_labels.append('1')
                            changed = True
                        elif clean_label == '1':
                            new_labels.append('0')
                            changed = True
                        else:
                            new_labels.append(clean_label)
                    
                    if changed:
                        lines[i] = '#Dx: ' + ','.join(new_labels) + '\n'
                        
        if changed:
            with open(filepath, 'w') as f:
                f.writelines(lines)
            return True
            
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        
    return False

if __name__ == "__main__":
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.hea'):
                if flip_labels(os.path.join(root, file)):
                    count += 1
    print(f"Successfully flipped labels in {count} .hea files in {directory} and its subfolders.")
