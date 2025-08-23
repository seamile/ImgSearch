#!/usr/bin/env python3
"""
iFinder command line interface
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

from PIL import Image

from .clip import Clip
from .storage import DB_DIR, VectorDatabase
from .utils import find_all_images, is_image, print_err


def main() -> None:  # noqa: C901
    """Main function for command line interface"""
    parser = ArgumentParser(prog='ifinder', description='iFinder - 图片查找工具')

    # Main arguments
    parser.add_argument('query', nargs='?', help='Search query (image path or keyword)')
    parser.add_argument('-a', '--add', nargs='+', metavar='PATH', help='Add images to index (file or directory path)')

    # Optional arguments
    parser.add_argument(
        '-d', '--database', type=Path, default=DB_DIR, help=f'Database directory path (default: {DB_DIR})'
    )
    parser.add_argument('-n', '--num', type=int, default=10, help='Number of search results (default: 10)')
    parser.add_argument('-m', '--model', type=str, help='CLIP model name')

    args = parser.parse_args()

    # Create instances
    imgbase = VectorDatabase(db_dir=args.database)
    clip = Clip(model_name=args.model) if args.model else Clip()

    try:
        # Add images to index
        if args.add:
            print('Building image index...')

            # Find all image files
            image_paths = list(find_all_images(args.add))
            if not image_paths:
                print_err('No image files found')
                sys.exit(1)

            added_count = 0
            batch_size = 32  # Process images in batches

            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i : i + batch_size]
                batch_images = []
                valid_paths = []

                # Load batch images
                for img_path in batch_paths:
                    try:
                        img = Image.open(img_path).convert('RGB')
                        batch_images.append(img)
                        valid_paths.append(img_path)
                    except Exception as e:
                        print_err(f'Failed to load image {img_path}: {e}')
                        continue

                if not batch_images:
                    continue

                try:
                    # Extract features
                    features = clip.embed_images(batch_images)

                    # Add to index (convert paths to strings)
                    labels = [str(path) for path in valid_paths]
                    imgbase.add_items(labels, features)
                    added_count += len(valid_paths)

                    print(f'Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images')

                except Exception as e:
                    print_err(f'Failed to process image batch: {e}')
                    continue

            if added_count > 0:
                print(f'Successfully added {added_count} images to index')
                imgbase.save()
                print(f'Index saved to {args.database}')
            else:
                print_err('No images added')
                sys.exit(1)

        # Search images
        elif args.query:
            print('Searching...')

            # Determine if query is image path or text
            query_path = Path(args.query)
            if query_path.exists() and is_image(query_path):
                # Image search
                try:
                    img = Image.open(query_path).convert('RGB')
                    feature = clip.embed_image(img)
                    results = imgbase.search(feature, k=args.num)
                except Exception as e:
                    print_err(f'Image search failed: {e}')
                    sys.exit(1)
            else:
                # Text search
                try:
                    feature = clip.embed_text(args.query)
                    results = imgbase.search(feature, k=args.num)
                except Exception as e:
                    print_err(f'Text search failed: {e}')
                    sys.exit(1)

            if results:
                print(f'\nFound {len(results)} similar results:')
                print('-' * 60)
                for i, (path, similarity) in enumerate(results, 1):
                    print(f'{i:2d}. {path} (similarity: {similarity}%)')
            else:
                print('No similar images found')

        else:
            parser.print_help()

    except KeyboardInterrupt:
        print('\nOperation cancelled')
        sys.exit(1)
    except Exception as e:
        print_err(f'Error occurred: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
