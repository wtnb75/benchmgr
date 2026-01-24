app do |env|
  [200, {}, ['']]
end

bind 'tcp://0.0.0.0:80'
