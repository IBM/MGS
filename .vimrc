"" Set indentation rule "
set tabstop=4
set softtabstop=4
set shiftwidth=4
set noexpandtab

"" Notify user if lines too long"
set colorcolumn=110
highlight ColorColumn ctermbg=darkgray

"" Vim search path "
:set path+=$NTSROOT/common/include,$NTSROOT/gsl/utils/std/include,$NTSROOT/gsl/framework/dca/include,$NTSROOT/mdl/include,$NTSROOT/gsl/framework/simulation/include,$NTSROOT/gsl/framework/parser/include,$NTSROOT/gsl/framework/factories/include,$NTSROOT/gsl/framework/networks/include,$NTSROOT/gsl/framework/dataitems/include,

" add ** to 'path' to enable recursive search
" using :find myfile.txt
set path+=**

" Build command by pressing F4 key
" NOTE: ! sign tell vim not jumping to the location of first error found
"nnoremap <F4> :make!<CR>
"

" ctags
:set tags=$NTSROOT/tags

au BufNewFile,BufRead *.di setlocal ft=d
au BufNewFile,BufRead *.gsl setlocal ft=gsl
au BufNewFile,BufRead *.mdl setlocal ft=mdl

