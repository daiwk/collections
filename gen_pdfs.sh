#echo "本文地址[https://github.com/daiwk/collections/blob/master/pdfs/collections.pdf](https://github.com/daiwk/collections/blob/master/pdfs/collections.pdf)\n" > collections.md

#cat ./posts/full.md | python3 trans_format.py >> ./pdfs/collections-pdf.md
cat ./posts/llm_aigc.md.raw | python3 ./posts/change_format_pdf.py > ./pdfs/1.llm_aigc-pdf.md
cat ./posts/llm_aigc.md.raw | python3 ./posts/change_format_md.py > ./posts/1.llm_aigc.md

cat ./posts/recommend.md.raw | python3 ./posts/change_format_pdf.py > ./pdfs/2.recommend-pdf.md
cat ./posts/recommend.md.raw | python3 ./posts/change_format_md.py > ./posts/2.recommend.md

cat ./posts/full.md.raw | python3 ./posts/change_format_pdf.py > ./pdfs/9.collections-pdf.md
cat ./posts/full.md.raw | python3 ./posts/change_format_md.py > ./posts/9.collections.md
cp ./posts/8.int-ml.md ./pdfs

python3 gen_dot_sub.py

##pandoc -N -s --toc --smart  --pdf-engine=xelatex -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in -f markdown+markdown_in_html_blocks+raw_html-implicit_figures ./collections-pdf.md -o ./pdfs/collections.pdf

cd pdfs
pandoc -N -s --toc --toc-depth=5 --pdf-engine=xelatex --columns=30 --standalone --html-q-tags -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in --metadata linkcolor=blue -f markdown+markdown_in_html_blocks+smart+raw_html-implicit_figures --highlight-style tango ./1.llm_aigc-pdf.md -o ./llm_aigc.pdf
pandoc -N -s --toc --toc-depth=5 --pdf-engine=xelatex --columns=30 --standalone --html-q-tags -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in --metadata linkcolor=blue -f markdown+markdown_in_html_blocks+smart+raw_html-implicit_figures --highlight-style tango ./2.recommend-pdf.md -o ./recommend.pdf
pandoc -N -s --toc --toc-depth=5 --pdf-engine=xelatex --columns=30 --standalone --html-q-tags -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in --metadata linkcolor=blue -f markdown+markdown_in_html_blocks+smart+raw_html-implicit_figures --highlight-style tango ./8.int-ml.md -o ./int-ml.pdf
pandoc -N -s --toc --toc-depth=5 --pdf-engine=xelatex --columns=30 --standalone --html-q-tags -V CJKmainfont='Heiti SC' -V mainfont='Times New Roman' -V geometry:margin=1in --metadata linkcolor=blue -f markdown+markdown_in_html_blocks+smart+raw_html-implicit_figures --highlight-style tango ./9.collections-pdf.md -o ./collections.pdf
cd -

rm -rf pdfs/*.md

git add ./assets
git add posts
git add pdfs
git add gen_pdfs.sh
git add gen_dot_sub.py
git commit -m 'x'
git push
